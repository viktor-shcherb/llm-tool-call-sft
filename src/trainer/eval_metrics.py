import gc
import hashlib
import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch

IGNORE_INDEX = -100
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


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


def _build_generation_prompt_str(
    tokenizer,
    context_messages: List[Dict[str, Any]],
    tools_shuffled: List[Dict[str, Any]],
) -> str:
    """
    Same as _build_generation_input but for text (for vLLM).
    """
    prompt = tokenizer.apply_chat_template(
        context_messages,
        tools=tools_shuffled,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    return prompt


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


def _aggregate_tool_metrics(
    per_example: List[Dict[str, Any]],
) -> Dict[str, float]:
    def _safe_mean(x):
        return float(sum(x) / len(x)) if x else 0.0

    all_prec = []
    all_rec = []
    all_f1 = []
    arg_exact = []
    arg_parse = []
    halluc = []
    num_tool_turns = 0
    num_no_tool_turns = 0

    for m in per_example:
        if m["kind"] == "tool":
            num_tool_turns += 1
            all_prec.append(m["precision"])
            all_rec.append(m["recall"])
            all_f1.append(m["f1"])
            arg_exact.extend(m["arg_exact_flags"])
            arg_parse.extend(m["arg_parse_flags"])
        else:
            num_no_tool_turns += 1
            halluc.append(m["hallucinated"])

    return {
        "tool_call_precision": _safe_mean(all_prec),
        "tool_call_recall": _safe_mean(all_rec),
        "tool_call_name_f1": _safe_mean(all_f1),
        "arg_exact_match_rate": _safe_mean(arg_exact),
        "arguments_parse_success_rate": _safe_mean(arg_parse),
        "no_tool_hallucination_rate": (1.0 - _safe_mean(halluc)) if halluc else 1.0,
        "num_tool_turns": num_tool_turns,
        "num_no_tool_turns": num_no_tool_turns,
    }


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


def eval_tool_calls(
    model_path: str,
    tokenizer,
    examples: List[ToolEvalTurn],
    global_tools: List[Dict[str, Any]],
    max_model_len: int,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
) -> Dict[str, float]:
    """
    Same metrics as eval_tool_calls, but generation is done by vLLM.
    Steps:
      - start vLLM on the saved model
      - build per-example prompts using the tokenizer chat template
      - generate
      - parse and score
      - destroy vLLM and free CUDA
    """
    from vllm import LLM, SamplingParams

    print('Starting vLLM setup')
    llm = LLM(
        model=model_path,
        tokenizer=model_path,  # reuse the saved tokenizer in the same dir
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    print('vLLM setup done!')

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts: List[str] = []
    example_meta: List[ToolEvalTurn] = []
    for ex in examples:
        shuffled_tools = _deterministic_shuffle_tools(global_tools, ex.session_id)
        prompt = _build_generation_prompt_str(
            tokenizer=tokenizer,
            context_messages=ex.context_messages,
            tools_shuffled=shuffled_tools,
        )
        prompts.append(prompt)
        example_meta.append(ex)

    print('Starting vLLM generation')
    outputs = llm.generate(prompts, sampling_params)
    print('vLLM generation done!')
    # vLLM returns results aligned to input order
    per_example_metrics: List[Dict[str, Any]] = []

    for ex, out in zip(example_meta, outputs):
        # pick first sequence
        generated_text = out.outputs[0].text

        pred_calls, parse_failures = _parse_predicted_tool_calls(generated_text)
        gold_calls = ex.gold_tool_calls

        if not ex.has_tools:
            hallucinated = 1 if len(pred_calls) > 0 else 0
            per_example_metrics.append(
                {
                    "kind": "no_tool",
                    "hallucinated": hallucinated,
                }
            )
            # still count malformed blocks
            continue

        gold_fn_names = [c["function"]["name"] for c in gold_calls]
        pred_fn_names = [c["function"]["name"] for c in pred_calls]

        prec, rec, f1 = _f1_precision_recall(gold_fn_names, pred_fn_names)

        # argument-level
        from collections import defaultdict, deque

        gold_by_fn = defaultdict(deque)
        for c in gold_calls:
            gold_by_fn[c["function"]["name"]].append(c["function"]["arguments"])

        arg_exact_flags: List[int] = []
        arg_parse_flags: List[int] = []

        for c in pred_calls:
            fn = c["function"]["name"]
            pred_args_norm = _normalize_args(c["function"]["arguments"])
            if pred_args_norm is None:
                arg_parse_flags.append(0)
                arg_exact_flags.append(0)
                continue

            arg_parse_flags.append(1)

            if gold_by_fn[fn]:
                gold_args_norm = _normalize_args(gold_by_fn[fn].popleft())
                if gold_args_norm is None:
                    arg_exact_flags.append(0)
                else:
                    arg_exact_flags.append(1 if pred_args_norm == gold_args_norm else 0)
            else:
                arg_exact_flags.append(0)

        # penalize malformed blocks
        for _ in range(parse_failures):
            arg_parse_flags.append(0)
            arg_exact_flags.append(0)

        per_example_metrics.append(
            {
                "kind": "tool",
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "arg_exact_flags": arg_exact_flags,
                "arg_parse_flags": arg_parse_flags,
            }
        )

    # teardown vLLM
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _aggregate_tool_metrics(per_example_metrics)
