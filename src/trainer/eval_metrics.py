import json
import re
import random
import hashlib
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

IGNORE_INDEX = -100


############################################
# helpers
############################################

def _unwrap_model(m):
    """
    Trainer may wrap the model in DDP.
    Return the underlying model.
    """
    return m.module if hasattr(m, "module") else m


############################################
# 1. Perplexity eval
############################################

@torch.inference_mode()
def eval_perplexity(
    model,
    dataset,
    batch_size: int,
    device: torch.device,
):
    """
    Compute mean loss over non-ignored tokens and perplexity.

    model: HF causal LM with .config.pad_token_id or .config.eos_token_id
    dataset: HF Dataset with columns:
        input_ids [T], attention_mask [T], labels [T]
    batch_size: eval batch size
    device: torch.device to run on
    """
    model = _unwrap_model(model)
    model.eval()
    model.to(device)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch],
                batch_first=True,
                padding_value=(
                    model.config.pad_token_id
                    if getattr(model.config, "pad_token_id", None) is not None
                    else model.config.eos_token_id
                ),
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [b["labels"] for b in batch],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
        },
    )

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dl, desc="eval_perplexity", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss  # scalar

        valid_count = (labels != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * valid_count
        total_tokens += valid_count

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_loss)))

    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "num_tokens": total_tokens,
    }


############################################
# 2. Tool-call eval
############################################
# Metrics:
# - precision / recall / F1 of function names when tools are expected
# - argument exact match rate
# - argument JSON parse success rate
# - hallucination rate when no tool call should be made


@dataclass
class ToolEvalTurn:
    session_id: str
    context_messages: List[Dict[str, Any]]
    gold_tool_calls: List[Dict[str, Any]]  # [] if none expected
    has_tools: bool  # True if gold_tool_calls non-empty


def _deterministic_shuffle_tools(
    tools: List[Dict[str, Any]],
    session_id: str,
):
    """
    Deterministically shuffle tool list using session_id.
    Matches training behavior where tool ordering is per-session stable.
    """
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)
    tools_copy = deepcopy(tools)
    rng.shuffle(tools_copy)
    return tools_copy


def prepare_tool_eval_examples(raw_sessions: List[Dict[str, Any]]) -> List[ToolEvalTurn]:
    """
    Build ToolEvalTurn objects for eval.

    raw_sessions[i] must be:
        {
            "session_id": str,
            "messages": [ ... ]  # chat messages including system/user/assistant/tool
        }

    For every assistant turn in each session:
    - context_messages is everything before that assistant turn
    - gold_tool_calls is assistant.tool_calls or []
    - has_tools is bool(tool_calls)

    Return flat list of ToolEvalTurn.
    """
    eval_examples: List[ToolEvalTurn] = []

    for sess in raw_sessions:
        session_id = sess["session_id"]
        msgs = sess["messages"]

        for idx, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue

            tool_calls = m.get("tool_calls", [])
            has_tools = bool(tool_calls)

            ctx = msgs[:idx]

            eval_examples.append(
                ToolEvalTurn(
                    session_id=session_id,
                    context_messages=ctx,
                    gold_tool_calls=deepcopy(tool_calls),
                    has_tools=has_tools,
                )
            )

    return eval_examples


############################################
# 2a. generation helpers
############################################

def _build_generation_input(
    tokenizer,
    context_messages: List[Dict[str, Any]],
    tools_shuffled: List[Dict[str, Any]],
) -> torch.LongTensor:
    """
    Build model input_ids for next-turn generation.
    We rely on add_generation_prompt=True so tokenizer will append
    the assistant preamble.
    """
    input_ids = tokenizer.apply_chat_template(
        context_messages,
        tools=tools_shuffled,
        tokenize=True,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
        return_tensors="pt",
    )
    # shape [1, seq_len]
    return input_ids


def _generate_assistant_turn(
    model,
    tokenizer,
    context_messages: List[Dict[str, Any]],
    tools_shuffled: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Generate assistant turn continuation for evaluation.

    Greedy decode if temperature == 0.0.
    """
    input_ids = _build_generation_input(
        tokenizer=tokenizer,
        context_messages=context_messages,
        tools_shuffled=tools_shuffled,
    ).to(device)

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=(temperature if temperature > 0.0 else None),
        pad_token_id=getattr(model.config, "pad_token_id", tokenizer.eos_token_id),
        eos_token_id=tokenizer.eos_token_id,
    )[0]

    # keep only generated continuation
    new_tokens = gen_ids[input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return text


############################################
# 2b. parsing tool calls from model output
############################################

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_predicted_tool_calls(
    generated_text: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract tool calls from generated_text.

    We expect spans like:
    <tool_call>{"function":{"name":"fn","arguments":{...}}}</tool_call>

    Returns:
      (predicted_calls, parse_failures)

    predicted_calls[i] = {
        "function": {
            "name": str,
            "arguments": <raw or json str>
        }
    }
    parse_failures increments for any block we could not json.loads.
    """
    predicted_calls: List[Dict[str, Any]] = []
    parse_failures = 0

    for block in _TOOL_CALL_BLOCK_RE.findall(generated_text):
        block_str = block.strip()
        try:
            call_obj = json.loads(block_str)
        except Exception:
            call_obj = None

        if call_obj is None:
            parse_failures += 1
            continue

        fn = call_obj.get("function", {})
        predicted_calls.append(
            {
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                }
            }
        )

    return predicted_calls, parse_failures


def _normalize_args(arg_val: Any):
    """
    Normalize predicted or gold arguments to dict[str,str].
    Returns None if parsing fails.

    - If arg_val is dict: stringify leaf values.
    - If arg_val is str: try json.loads then same.
    - Else: None.
    """
    if isinstance(arg_val, dict):
        obj = arg_val
    elif isinstance(arg_val, str):
        try:
            obj = json.loads(arg_val)
        except Exception:
            return None
    else:
        return None

    if not isinstance(obj, dict):
        return None

    return {k: str(v).strip() for k, v in obj.items()}


############################################
# 2c. metric math utilities
############################################

def _f1_precision_recall(
    true_fns: List[str],
    pred_fns: List[str],
) -> Tuple[float, float, float]:
    """
    Micro precision, recall, F1 over function name multisets.
    """
    from collections import Counter
    gold_ct = Counter(true_fns)
    pred_ct = Counter(pred_fns)

    tp = sum(min(gold_ct[fn], pred_ct[fn]) for fn in pred_ct)

    fp = sum(
        max(pred_ct[fn] - gold_ct.get(fn, 0), 0)
        for fn in pred_ct
    )
    fn = sum(
        max(gold_ct[fn] - pred_ct.get(fn, 0), 0)
        for fn in gold_ct
    )

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


############################################
# 2d. full tool-call eval
############################################

@torch.inference_mode()
def eval_tool_calls(
    model,
    tokenizer,
    examples: List[ToolEvalTurn],
    global_tools: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    """
    Evaluate tool-calling behavior.

    Inputs:
      model, tokenizer: HF model/tokenizer
      examples: list[ToolEvalTurn]
      global_tools: list of tool specs
      device: model device
      max_new_tokens: generation cap
      temperature: decoding temperature

    Returns dict with:
      tool_call_precision
      tool_call_recall
      tool_call_name_f1
      arg_exact_match_rate
      arguments_parse_success_rate
      no_tool_hallucination_rate
      num_tool_turns
      num_no_tool_turns
    """
    model = _unwrap_model(model)
    model.eval()
    model.to(device)

    all_prec: List[float] = []
    all_rec: List[float] = []
    all_f1: List[float] = []
    arg_exact_flags: List[int] = []
    arg_parse_success_flags: List[int] = []
    hallucinated_flags: List[int] = []

    for ex in tqdm(examples, desc="eval_tool_calls", leave=False):
        shuffled_tools = _deterministic_shuffle_tools(
            global_tools,
            ex.session_id,
        )

        gen_text = _generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            context_messages=ex.context_messages,
            tools_shuffled=shuffled_tools,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        pred_calls, parse_failures = _parse_predicted_tool_calls(gen_text)
        gold_calls = ex.gold_tool_calls

        # hallucination metric:
        # if gold expects no tools then model should emit none
        if not ex.has_tools:
            hallucinated = 1 if len(pred_calls) > 0 else 0
            hallucinated_flags.append(hallucinated)

        # compute precision / recall / F1 and argument metrics
        if ex.has_tools:
            gold_fn_names = [c["function"]["name"] for c in gold_calls]
            pred_fn_names = [c["function"]["name"] for c in pred_calls]

            prec, rec, f1 = _f1_precision_recall(
                gold_fn_names,
                pred_fn_names,
            )
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)

            # argument match and parse success
            from collections import defaultdict, deque
            gold_by_fn = defaultdict(deque)
            for c in gold_calls:
                gold_by_fn[c["function"]["name"]].append(
                    c["function"]["arguments"]
                )

            for c in pred_calls:
                fn = c["function"]["name"]
                pred_args_norm = _normalize_args(
                    c["function"]["arguments"]
                )
                if pred_args_norm is None:
                    arg_parse_success_flags.append(0)
                    arg_exact_flags.append(0)
                    continue
                arg_parse_success_flags.append(1)

                if gold_by_fn[fn]:
                    gold_args_norm = _normalize_args(
                        gold_by_fn[fn].popleft()
                    )
                    if gold_args_norm is None:
                        arg_exact_flags.append(0)
                    else:
                        arg_exact_flags.append(
                            1 if pred_args_norm == gold_args_norm else 0
                        )
                else:
                    arg_exact_flags.append(0)

        # penalize malformed tool_call blocks
        for _ in range(parse_failures):
            arg_parse_success_flags.append(0)
            arg_exact_flags.append(0)

    def _safe_mean(arr: List[float]) -> float:
        return float(sum(arr) / len(arr)) if len(arr) > 0 else 0.0

    metrics = {
        "tool_call_precision": _safe_mean(all_prec),
        "tool_call_recall": _safe_mean(all_rec),
        "tool_call_name_f1": _safe_mean(all_f1),
        "arg_exact_match_rate": _safe_mean(arg_exact_flags),
        "arguments_parse_success_rate": _safe_mean(
            arg_parse_success_flags
        ),
        "no_tool_hallucination_rate": (
            1.0 - _safe_mean(hallucinated_flags)
            if len(hallucinated_flags) > 0
            else 1.0
        ),
        "num_tool_turns": len(all_f1),
        "num_no_tool_turns": len(hallucinated_flags),
    }

    return metrics


__all__ = [
    "IGNORE_INDEX",
    "ToolEvalTurn",
    "prepare_tool_eval_examples",
    "eval_perplexity",
    "eval_tool_calls",
]
