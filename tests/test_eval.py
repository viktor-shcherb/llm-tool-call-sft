import torch
import math
import pytest
from eval import (
    eval_perplexity,
    prepare_tool_eval_examples,
    eval_tool_calls,
    _parse_predicted_tool_calls,
    _f1_precision_recall,
)
from types import SimpleNamespace


def test_eval_perplexity_smoke(tiny_model_and_tokenizer, toy_tokenized_dataset):
    """
    Sanity check:
    - eval_perplexity runs a real forward pass on CPU
    - returns finite loss and perplexity
    """
    model, tok = tiny_model_and_tokenizer
    device = torch.device("cpu")
    out = eval_perplexity(
        model=model,
        dataset=toy_tokenized_dataset,
        batch_size=2,
        device=device,
    )

    assert "loss" in out
    assert "perplexity" in out
    assert out["num_tokens"] > 0
    assert math.isfinite(out["loss"])
    assert out["perplexity"] > 0
    assert math.isfinite(out["perplexity"])


def test__parse_predicted_tool_calls_roundtrip():
    """
    Parser should extract function name and raw arguments string
    from a valid <tool_call>...</tool_call> block.
    """
    text = (
        "<tool_call>"
        '{"id":"call_1","type":"function",'
        '"function":{"name":"lookup_inventory","arguments":"{\\"vin\\": \\"123\\"}"}}'
        "</tool_call>"
    )

    calls, fails = _parse_predicted_tool_calls(text)
    assert fails == 0
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_inventory"
    assert '"vin"' in calls[0]["function"]["arguments"]


def test__f1_precision_recall_exact_match():
    """
    When pred == gold the F1 should be 1.
    """
    prec, rec, f1 = _f1_precision_recall(
        ["lookup_inventory", "schedule_test_drive"],
        ["lookup_inventory", "schedule_test_drive"],
    )
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_tool_eval_metrics_perfect_match(
    monkeypatch,
    tiny_model_and_tokenizer,
    raw_sessions_fixture,
    global_tools_fixture,
):
    """
    Uses monkeypatch to force ideal generations.
    Confirms metric math gives 1.0 when model is perfect.
    """

    from eval import eval_tool_calls, prepare_tool_eval_examples

    # Build eval examples from the raw sessions
    examples = prepare_tool_eval_examples(
        raw_sessions_fixture,
        global_tools_fixture,
    )

    # gold-like generations
    tool_call_generation = (
        "<tool_call>"
        '{"id":"call_1","type":"function",'
        '"function":{"name":"lookup_inventory","arguments":"{\\"vin\\": \\"123\\"}"}}'
        "</tool_call>"
    )
    summary_generation = (
        "Yes. The Civic with VIN 123 is in stock and listed at $15,999."
    )
    greeting_generation = (
        "Hello. How can I help you with a vehicle today?"
    )

    def fake_generate_assistant_turn(
        model,
        tokenizer,
        context_messages,
        tools_shuffled,
        device,
        max_new_tokens=256,
        temperature=0.0,
    ):
        """
        Policy:
        - If shopper asked about a VIN AND we have NOT yet seen a tool response,
          return a <tool_call> block.
        - If shopper asked about a VIN AND we HAVE already seen a tool response
          (role == 'tool' present in context), return summary_generation.
        - If shopper just said hi, return greeting_generation.
        """

        # detect if this is after tool execution
        saw_tool_response = any(m["role"] == "tool" for m in context_messages)

        # last user utterance drives branch between VIN path vs greeting path
        last_user_msg = [m for m in context_messages if m["role"] == "user"][-1]["content"]

        vin_query = ("VIN is 123" in last_user_msg) or ("Follow the policy" in last_user_msg)

        if vin_query and not saw_tool_response:
            # first assistant turn: should request lookup_inventory
            return tool_call_generation

        if vin_query and saw_tool_response:
            # second assistant turn: should summarize results, no tool call
            return summary_generation

        # else it's the greeting session
        return greeting_generation

    # patch eval._generate_assistant_turn so eval_tool_calls uses our deterministic stub
    monkeypatch.setattr(
        "eval._generate_assistant_turn",
        fake_generate_assistant_turn,
        raising=True,
    )

    model, tok = tiny_model_and_tokenizer
    device = torch.device("cpu")

    metrics = eval_tool_calls(
        model=model,
        tokenizer=tok,
        examples=examples,
        global_tools=global_tools_fixture,
        device=device,
        max_new_tokens=1024,
        temperature=0.0,
    )

    assert metrics["tool_call_name_f1"] == 1.0
    assert metrics["tool_call_precision"] == 1.0
    assert metrics["tool_call_recall"] == 1.0
    assert metrics["arg_exact_match_rate"] == 1.0
    assert metrics["arguments_parse_success_rate"] == 1.0
    assert metrics["no_tool_hallucination_rate"] == 1.0
    assert metrics["num_tool_turns"] >= 1
    assert metrics["num_no_tool_turns"] >= 1



def test_tool_eval_metrics_real_model_smoke(
    tiny_model_and_tokenizer,
    raw_sessions_fixture,
    global_tools_fixture,
):
    """
    Real integration test.

    This test:
    - builds eval examples from real conversations with tool_calls
    - runs eval_tool_calls using the actual model.generate() from smollm2-135m-instruct
    - does NOT monkeypatch generation
    - asserts eval_tool_calls returns numeric metrics and does not crash
    """

    model, tok = tiny_model_and_tokenizer
    device = torch.device("cpu")

    # build eval examples from fixture sessions
    examples = prepare_tool_eval_examples(
        raw_sessions_fixture,
        global_tools_fixture,
    )

    # run eval_tool_calls end to end with real .generate()
    metrics = eval_tool_calls(
        model=model,
        tokenizer=tok,
        examples=examples,
        global_tools=global_tools_fixture,
        device=device,
        max_new_tokens=1024,
        temperature=0.0,       # greedy
    )
    print(metrics)

    # basic sanity on keys
    expected_keys = {
        "tool_call_precision",
        "tool_call_recall",
        "tool_call_name_f1",
        "arg_exact_match_rate",
        "arguments_parse_success_rate",
        "no_tool_hallucination_rate",
        "num_tool_turns",
        "num_no_tool_turns",
    }
    assert expected_keys.issubset(set(metrics.keys()))

    # metrics should be finite numbers in [0,1] where applicable
    bounded_keys = [
        "tool_call_precision",
        "tool_call_recall",
        "tool_call_name_f1",
        "arg_exact_match_rate",
        "arguments_parse_success_rate",
        "no_tool_hallucination_rate",
    ]
    for k in bounded_keys:
        v = metrics[k]
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0

    assert metrics["tool_call_name_f1"] == 1.0
    assert metrics["tool_call_precision"] == 1.0
    assert metrics["tool_call_recall"] == 1.0
    assert metrics["arg_exact_match_rate"] == 1.0
    assert metrics["arguments_parse_success_rate"] == 1.0
    assert metrics["no_tool_hallucination_rate"] == 1.0
    assert metrics["num_tool_turns"] >= 1
    assert metrics["num_no_tool_turns"] >= 1
