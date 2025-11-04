import json
import re
import random
import hashlib
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

IGNORE_INDEX = -100


############################################
# helpers
############################################

def _is_fsdp(m) -> bool:
    """
    Return True for torch.distributed.fsdp.FullyShardedDataParallel
    without importing it explicitly.
    """
    cls = m.__class__.__name__
    module = m.__class__.__module__
    return cls == "FullyShardedDataParallel" or "fsdp" in module.lower()


def _unwrap_model(m):
    """
    Unwrap DDP-like wrappers to get to the HF model for config access.
    For FSDP we return the wrapper itself because forward/generate must
    go through it.
    """
    if _is_fsdp(m):
        return m
    return m.module if hasattr(m, "module") else m


def _unwrap_for_config(m):
    # for DDP: return .module
    # for FSDP: we have to read config from the wrapped module if present
    if _is_fsdp(m):
        # FSDP usually stores the real module in .module
        return getattr(m, "module", m)
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
    # distributed env info
    import torch.distributed as dist
    dist_on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_on else 0
    world_size = dist.get_world_size() if dist_on else 1

    exec_model = model
    base_model = _unwrap_for_config(model)

    exec_model.eval()

    # only move plain model
    if not dist_on and not _is_fsdp(exec_model) and not hasattr(exec_model, "module"):
        exec_model.to(device)

    pad_id = getattr(base_model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = base_model.config.eos_token_id

    # shard dataset over ranks
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if dist_on
        else None
    )

    def _collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            ),
        }

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        collate_fn=_collate,
    )

    local_loss_sum = 0.0
    local_token_sum = 0

    for batch in tqdm(
        dl,
        desc="eval_perplexity",
        leave=False,
        disable=(rank != 0),
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # optional speedup on GPUs
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = exec_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss

        valid_count = (labels != IGNORE_INDEX).sum().item()
        local_loss_sum += loss.item() * valid_count
        local_token_sum += valid_count

    # aggregate across ranks
    loss_tokens = torch.tensor(
        [local_loss_sum, float(local_token_sum)],
        device=device,
        dtype=torch.float64,
    )
    if dist_on:
        dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM)

    total_loss = float(loss_tokens[0].item())
    total_tokens = int(loss_tokens[1].item())

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_loss)))

    # everyone returns the same dict so HF Trainer is happy
    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "num_tokens": total_tokens,
    }


############################################
# 2. Tool-call eval
############################################
# (unchanged parts omitted where possible)
############################################

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
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)
    tools_copy = deepcopy(tools)
    rng.shuffle(tools_copy)
    return tools_copy


def prepare_tool_eval_examples(raw_sessions: List[Dict[str, Any]]) -> List[ToolEvalTurn]:
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


def _build_generation_input(
    tokenizer,
    context_messages: List[Dict[str, Any]],
    tools_shuffled: List[Dict[str, Any]],
) -> torch.LongTensor:
    input_ids = tokenizer.apply_chat_template(
        context_messages,
        tools=tools_shuffled,
        tokenize=True,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
        return_tensors="pt",
    )
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
    input_ids = _build_generation_input(
        tokenizer=tokenizer,
        context_messages=context_messages,
        tools_shuffled=tools_shuffled,
    ).to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # call the (possibly FSDP) model directly
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=(temperature if temperature > 0.0 else None),
        pad_token_id=getattr(model.config, "pad_token_id", tokenizer.eos_token_id),
        eos_token_id=tokenizer.eos_token_id,
    )[0]

    new_tokens = gen_ids[input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return text


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_predicted_tool_calls(
    generated_text: str,
) -> Tuple[List[Dict[str, Any]], int]:
    predicted_calls: List[Dict[str, Any]] = []
    parse_failures = 0

    for block in _TOOL_CALL_BLOCK_RE.findall(generated_text):
        block_str = block.strip()
        try:
            call_obj = json.loads(block_str)
        except Exception:
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


def _f1_precision_recall(
    true_fns: List[str],
    pred_fns: List[str],
) -> Tuple[float, float, float]:
    from collections import Counter
    gold_ct = Counter(true_fns)
    pred_ct = Counter(pred_fns)

    tp = sum(min(gold_ct[fn], pred_ct[fn]) for fn in pred_ct)

    fp = sum(max(pred_ct[fn] - gold_ct.get(fn, 0), 0) for fn in pred_ct)
    fn = sum(max(gold_ct[fn] - pred_ct.get(fn, 0), 0) for fn in gold_ct)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


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
    Same logic, but:
    - do not .to(device) an FSDP/DDP model
    - still access config via unwrapped model
    """
    exec_model = model
    base_model = _unwrap_model(model)

    exec_model.eval()

    if not _is_fsdp(exec_model) and not hasattr(exec_model, "module"):
        exec_model.to(device)

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
            model=exec_model,
            tokenizer=tokenizer,
            context_messages=ex.context_messages,
            tools_shuffled=shuffled_tools,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        pred_calls, parse_failures = _parse_predicted_tool_calls(gen_text)
        gold_calls = ex.gold_tool_calls

        if not ex.has_tools:
            hallucinated = 1 if len(pred_calls) > 0 else 0
            hallucinated_flags.append(hallucinated)

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
