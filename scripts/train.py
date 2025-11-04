import argparse
import importlib
from typing import Any, Dict

import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainingArguments

from trainer.lora_model import build_model_and_tokenizer
from trainer.collate import make_causal_lm_collate_fn
from trainer.tool_trainer import ToolTrainer

from dotenv import load_dotenv
load_dotenv()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _load_data_module(cfg: Dict[str, Any]):
    mod_name = cfg["data"]["load_module"]
    return importlib.import_module(mod_name)


def _build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    return TrainingArguments(**cfg["train"])


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    _set_seed(cfg["train"]["seed"])

    model, tokenizer = build_model_and_tokenizer(cfg)

    data_mod = _load_data_module(cfg)

    (
        train_dataset,
        eval_dataset,
        tool_eval_examples,
        global_tools,
        max_ctx,
    ) = data_mod.build_datasets(cfg)

    data_collator = make_causal_lm_collate_fn(
        pad_token_id=tokenizer.pad_token_id,
    )

    training_args = _build_training_args(cfg)

    trainer = ToolTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        tokenizer_for_tools=tokenizer,
        tool_eval_examples=tool_eval_examples,
        global_tools=global_tools,
        max_new_tokens_eval=cfg["eval"]["max_new_tokens_eval"],
        temperature_eval=cfg["eval"]["temperature"],
        n_short_eval_examples=cfg["data"]["n_tool_sessions_eval"],
    )

    trainer.evaluate(eval_dataset, metric_key_prefix="full_eval")

    trainer.train()

    trainer.save_model()
    trainer.evaluate(eval_dataset, metric_key_prefix="full_eval")
    trainer.push_to_hub(commit_message="train: finish")


if __name__ == "__main__":
    main()
