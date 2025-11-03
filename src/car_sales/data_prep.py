from typing import Dict, Any, Tuple, List, Optional
from functools import partial

import random
import hashlib
from copy import deepcopy

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

from car_sales.prompt import (
    load_sysprompt_text,
    load_global_tools,
    session_to_messages,
)
from trainer.eval_metrics import prepare_tool_eval_examples, IGNORE_INDEX



def _deterministic_shuffle_tools(
    tools: List[Dict[str, Any]],
    session_id: str,
) -> List[Dict[str, Any]]:
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)

    tools_copy = deepcopy(tools)
    rng.shuffle(tools_copy)
    return tools_copy


def tokenize_with_role_spans(
    messages: List[Dict[str, Any]],
    session_id: str,
    tokenizer,
    global_tools: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    all_input_ids: List[int] = []
    all_labels: List[int] = []
    prev_input_ids = None

    shuffled_tools = _deterministic_shuffle_tools(global_tools, session_id)

    for i in range(len(messages)):
        prefix_ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            tools=shuffled_tools,
            tokenize=True,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
            return_tensors=None,
        )

        if prev_input_ids is None:
            new_ids = prefix_ids
        else:
            new_ids = prefix_ids[len(prev_input_ids):]

        role_i = messages[i]["role"]

        if role_i == "assistant":
            role_labels = new_ids[:]
        else:
            role_labels = [IGNORE_INDEX] * len(new_ids)

        all_input_ids.extend(new_ids)
        all_labels.extend(role_labels)
        prev_input_ids = prefix_ids

    attn_mask = [1] * len(all_input_ids)

    return {
        "input_ids": torch.as_tensor(all_input_ids),
        "labels": torch.as_tensor(all_labels),
        "attention_mask": torch.as_tensor(attn_mask),
    }


def _preprocess_single_example(
    example: Dict[str, Any],
    *,
    tokenizer,
    base_sysprompt_text: str,
    timezone_str: str,
    global_tools: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    messages = session_to_messages(
        example=example,
        base_sysprompt_text=base_sysprompt_text,
        timezone_str=timezone_str,
    )
    feats = tokenize_with_role_spans(
        messages=messages,
        session_id=example["session_id"],
        tokenizer=tokenizer,
        global_tools=global_tools,
    )
    return feats


def within_ctx(example: Dict[str, Any], max_ctx: int) -> bool:
    return len(example["input_ids"]) <= max_ctx


def _build_tool_eval_examples(
    raw_test_split: Dataset,
    base_sysprompt_text: str,
    timezone_str: str,
    n_sessions: Optional[int],
):
    raw_sessions = []
    n_sessions = n_sessions or len(raw_test_split)
    limit = min(n_sessions, len(raw_test_split))

    for i in range(limit):
        row = raw_test_split[i]
        msgs = session_to_messages(
            example=row,
            base_sysprompt_text=base_sysprompt_text,
            timezone_str=timezone_str,
        )
        raw_sessions.append(
            {
                "session_id": row["session_id"],
                "messages": msgs,
            }
        )

    tool_eval_examples = prepare_tool_eval_examples(raw_sessions=raw_sessions)

    return tool_eval_examples


def build_datasets(
    cfg: Dict[str, Any],
) -> Tuple[Dataset, Dataset, List[Any], List[Any], List[Dict[str, Any]], int]:
    """
    End-to-end dataset prep for this domain.

    cfg is the full YAML (already loaded).
    We only read cfg["data"][...].

    Steps
    1. load dataset
    2. load tokenizer, system prompt text, tool specs
    3. map to tokenized tensors with role-masked labels
    4. drop long sequences if requested
    5. build tool_eval_examples

    Returns
      train_dataset (tokenized + filtered)
      eval_dataset (tokenized + filtered)
      tool_eval_examples (list[ToolEvalTurn])
      global_tools (tool schemas)
      max_ctx (int)
    """

    data_cfg = cfg["data"]

    # 1. dataset
    ds: DatasetDict = load_dataset(data_cfg["dataset_name"])

    # 2. assets
    timezone_str = data_cfg["timezone"]
    base_sysprompt_text = load_sysprompt_text(data_cfg["sysprompt_path"])
    global_tools = load_global_tools(data_cfg["tools_path"])

    # tokenizer local to data prep
    tokenizer_name = data_cfg["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # suppress "Token indices sequence length is longer than ..." spam
    # we control overflow ourselves via drop_oversized
    tokenizer.model_max_length = int(data_cfg["max_context_length"])

    # 3. map preprocessing
    preprocess_fn = partial(
        _preprocess_single_example,
        tokenizer=tokenizer,
        base_sysprompt_text=base_sysprompt_text,
        timezone_str=timezone_str,
        global_tools=global_tools,
    )

    train_proc = ds["train"].map(
        preprocess_fn,
        remove_columns=ds["train"].column_names,
        desc="preprocess train",
    )

    test_proc = ds["test"].map(
        preprocess_fn,
        remove_columns=ds["test"].column_names,
        desc="preprocess test",
    )

    # cast to torch format
    train_proc.set_format(
        type="torch",
        columns=["input_ids", "labels", "attention_mask"],
    )
    test_proc.set_format(
        type="torch",
        columns=["input_ids", "labels", "attention_mask"],
    )

    # 4. context filtering
    max_ctx = int(data_cfg["max_context_length"])
    max_gen = int(cfg["eval"]["max_new_tokens_eval"])

    if data_cfg["drop_oversized"]:
        train_proc = train_proc.filter(
            lambda ex: within_ctx(ex, max_ctx),
            desc="filter train oversized",
        )
        test_proc = test_proc.filter(
            lambda ex: within_ctx(ex, max_ctx - max_gen),
            desc="filter test oversized",
        )

    # 5. build tool_eval_examples from raw (non-tokenized) test split
    tool_eval_examples = _build_tool_eval_examples(
        raw_test_split=ds["test"],
        base_sysprompt_text=base_sysprompt_text,
        timezone_str=timezone_str,
        n_sessions=None,
    )
    tool_eval_examples_short = random.sample(
        tool_eval_examples,
        k=min(data_cfg["n_tool_sessions_eval"], len(tool_eval_examples)),
    )

    return (
        train_proc,
        test_proc,
        tool_eval_examples,
        tool_eval_examples_short,
        global_tools,
        max_ctx,
    )
