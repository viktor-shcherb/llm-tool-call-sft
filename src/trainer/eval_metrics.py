import hashlib
import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI

IGNORE_INDEX = -100
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class ToolEvalTurn:
    session_id: str
    context_messages: List[Dict[str, Any]]
    gold_tool_calls: List[Dict[str, Any]]
    has_tools: bool


def _deterministic_shuffle_tools(tools: List[Dict[str, Any]], session_id: str):
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)
    tools_copy = deepcopy(tools)
    rng.shuffle(tools_copy)
    return tools_copy


def _parse_predicted_tool_calls(generated_text: str) -> Tuple[List[Dict[str, Any]], int]:
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


def _aggregate_tool_metrics(per_example: List[Dict[str, Any]]) -> Dict[str, float]:
    def _safe_mean(x):
        return float(sum(x) / len(x)) if x else 0.0

    all_prec, all_rec, all_f1 = [], [], []
    arg_exact, arg_parse, halluc = [], [], []
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


def _f1_precision_recall(true_fns: List[str], pred_fns: List[str]) -> Tuple[float, float, float]:
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


def _openaiify_messages(messages):
    """Ensure assistant tool_call messages match OpenAI chat schema."""
    out = []
    for m in messages:
        m = deepcopy(m)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            fixed_calls = []
            for tc in tool_calls:
                tc = deepcopy(tc)
                # required by OpenAI schema
                tc.setdefault("type", "function")

                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    # convert dict -> json string
                    fn["arguments"] = json.dumps(args)
                tc["function"] = fn
                fixed_calls.append(tc)

            m["tool_calls"] = fixed_calls
            # some datasets omit content for tool-calling assistant messages
            m.setdefault("content", "")
        out.append(m)
    return out


def eval_tool_calls(
    client: OpenAI,
    model: str,
    tokenizer,
    examples: List[ToolEvalTurn],
    global_tools: List[Dict[str, Any]],
    max_model_len: int,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    lora_name: Optional[str] = None,
) -> Dict[str, float]:
    per_example_metrics: List[Dict[str, Any]] = []

    for ex in examples:
        shuffled_tools = _deterministic_shuffle_tools(global_tools, ex.session_id)
        messages = _openaiify_messages(ex.context_messages)

        extra_body = {}
        if lora_name:
            model = lora_name

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=shuffled_tools,
                max_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )
        except Exception as e:
            print(f"Error evaluating tool calls: {e}")
            continue

        choice = resp.choices[0].message
        generated_text = choice.content or ""

        pred_calls, parse_failures = _parse_predicted_tool_calls(generated_text)
        gold_calls = ex.gold_tool_calls

        if not ex.has_tools:
            hallucinated = 1 if len(pred_calls) > 0 else 0
            per_example_metrics.append({"kind": "no_tool", "hallucinated": hallucinated})
            continue

        gold_fn_names = [c["function"]["name"] for c in gold_calls]
        pred_fn_names = [c["function"]["name"] for c in pred_calls]
        prec, rec, f1 = _f1_precision_recall(gold_fn_names, pred_fn_names)

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

    return _aggregate_tool_metrics(per_example_metrics)
