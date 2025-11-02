import types
import torch
from datasets import Dataset, DatasetDict
import pytest
from copy import deepcopy

import car_sales.data_prep as dp


class DummyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 7
    pad_token = "<eos>"
    pad_token_id = 7

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        chat_template_kwargs=None,
        return_tensors=None,
    ):
        # Each message adds 5 tokens
        n = len(messages)
        return list(range(n * 5))


class LongTokenizer(DummyTokenizer):
    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        chat_template_kwargs=None,
        return_tensors=None,
    ):
        # Each message adds 100 tokens
        n = len(messages)
        return list(range(n * 100))


def _make_ds_row(session_id):
    return {
        "session_id": session_id,
        "inferred_date": "2024-03-10",
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
    }


@pytest.fixture
def fake_cfg():
    return {
        "data": {
            "load_module": "car_sales",
            "dataset_name": "does/not/matter",
            "timezone": "Europe/Zurich",
            "sysprompt_path": "data/sysprompt.md",
            "tools_path": "data/tool_calls.json",
            "tokenizer": "viktoroo/SmolLM2-360M-Tools",
            "max_context_length": 12,
            "drop_oversized": True,
            "n_tool_sessions_eval": 4,
        }
    }


def test_build_datasets_filters_oversized(monkeypatch, fake_cfg):
    # custom session_to_messages to control sequence length per row
    # "long" -> 2 messages -> 200 tokens with LongTokenizer
    # "short" -> 1 message -> 100 tokens with LongTokenizer
    def _session_to_messages_len_by_id(example, base_sysprompt_text, timezone_str):
        if example["session_id"] == "long":
            return [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        else:  # "short"
            return [
                {"role": "user", "content": "hi"},
            ]

    # dataset with one long row and one short row in train
    long_row = _make_ds_row("long")
    short_row = _make_ds_row("short")

    raw_train = Dataset.from_list([long_row, short_row])
    raw_test = Dataset.from_list([short_row])

    monkeypatch.setattr(
        dp,
        "load_dataset",
        lambda name: DatasetDict({"train": raw_train, "test": raw_test}),
    )

    monkeypatch.setattr(
        dp,
        "load_sysprompt_text",
        lambda path: "# Section\n- a\n- b\n"
    )
    monkeypatch.setattr(
        dp,
        "load_global_tools",
        lambda path: [
            {"type": "function", "function": {"name": "lookup", "parameters": {}}},
        ],
    )
    monkeypatch.setattr(
        dp,
        "session_to_messages",
        _session_to_messages_len_by_id,
    )

    # tokenizer now is LongTokenizer (100 tokens per message)
    monkeypatch.setattr(
        dp,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda name: LongTokenizer()),
    )

    # adjust cfg so that:
    #   short (1 msg ~100 tokens) <= max_ctx
    #   long  (2 msg ~200 tokens) > max_ctx
    cfg_local = deepcopy(fake_cfg)
    cfg_local["data"]["max_context_length"] = 150  # threshold between 100 and 200

    train_ds, eval_ds, tool_eval_examples, global_tools, max_ctx = dp.build_datasets(cfg_local)

    # long_row should be dropped, short_row kept
    assert len(train_ds) == 1
    # eval_ds only had "short" anyway so still 1
    assert len(eval_ds) == 1

    # confirm surviving row is "short"
    # note: after filtering HF Dataset reorders indices, so just assert length of tensors:
    sample = train_ds[0]
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert sample["input_ids"].shape[0] <= max_ctx

    # max_ctx matches cfg_local after override
    assert max_ctx == cfg_local["data"]["max_context_length"]
