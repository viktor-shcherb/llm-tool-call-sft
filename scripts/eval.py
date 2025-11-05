#!/usr/bin/env python3
"""
scripts/eval.py

Run tool-calling eval with vLLM against the model published to the hub.

Usage:
    python scripts/eval.py --config configs/car_sales/qwen3-4b.yaml \
        [--tensor-parallel-size 1] [--max-model-len 16384]

Requirements:
    - HF token in env if repo is private:  HF_TOKEN=...
    - vLLM installed
"""

import argparse
import importlib
from typing import Any, Dict

import yaml
from transformers import AutoTokenizer

from trainer.eval_metrics import eval_tool_calls

import dotenv
dotenv.load_dotenv()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_module(cfg: Dict[str, Any]):
    mod_name = cfg["data"]["load_module"]
    return importlib.import_module(mod_name)


def pick_model_id(cfg: Dict[str, Any]) -> str:
    # per your request: prefer hub_model_id from train
    train_cfg = cfg.get("train", {})
    hub_id = train_cfg.get("hub_model_id")
    if hub_id:
        return hub_id
    # fallback to base model
    model_cfg = cfg.get("model", {})
    base_id = (
        model_cfg.get("base_model_name")
        or model_cfg.get("base_model_name_or_path")
        or model_cfg.get("name_or_path")
    )
    if not base_id:
        raise ValueError("Cannot determine model id; set train.hub_model_id or model.base_model_name")
    return base_id


def pick_tokenizer_id(cfg: Dict[str, Any], model_id: str) -> str:
    data_cfg = cfg.get("data", {})
    tok_id = data_cfg.get("tokenizer")
    return tok_id or model_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 1) figure out model + tokenizer
    model_id = pick_model_id(cfg)
    tokenizer_id = pick_tokenizer_id(cfg, model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False, trust_remote_code=True)

    # 2) build tool-eval examples and global tools from the same data module as training
    data_mod = load_data_module(cfg)
    (
        _train_dataset,
        _eval_dataset,
        tool_eval_examples,
        global_tools,
        max_ctx_from_data,
    ) = data_mod.build_datasets(cfg)

    # 3) eval settings
    eval_cfg = cfg.get("eval", {})
    max_new_tokens = int(eval_cfg.get("max_new_tokens_eval", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))

    max_model_len = args.max_model_len or int(cfg["data"].get("max_context_length", max_ctx_from_data))

    # 4) run vLLM-based eval
    metrics = eval_tool_calls(
        model_path=model_id,
        tokenizer=tokenizer,
        examples=tool_eval_examples,
        global_tools=global_tools,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 5) print result as flat JSON-ish
    # do not pretty-print too much; simple stdout is fine
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    # make sure HF token is used if set
    # vLLM will pick up HF_TOKEN / HUGGING_FACE_HUB_TOKEN from env
    main()
