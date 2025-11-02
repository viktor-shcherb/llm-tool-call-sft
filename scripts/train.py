import sys
import pathlib
import argparse
import importlib
from typing import Any, Dict

import yaml
import random
import numpy as np
import torch
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
    """
    Dynamically import data module based on cfg["data"]["load_module"].
    That module must expose build_datasets(cfg).
    """
    mod_name = cfg["data"]["load_module"]
    return importlib.import_module(mod_name)


def _build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    """
    Map YAML -> transformers.TrainingArguments.
    """
    run_cfg = cfg["run"]
    train_cfg = cfg["train"]

    training_args = TrainingArguments(
        output_dir=run_cfg["output_dir"],
        overwrite_output_dir=train_cfg["overwrite_output_dir"],

        # precision
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],

        # batch + schedule
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],

        # logging / eval / save
        logging_strategy="steps",
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],

        # distributed
        ddp_find_unused_parameters=train_cfg["ddp_find_unused_parameters"],

        # misc
        remove_unused_columns=train_cfg["remove_unused_columns"],
        report_to=run_cfg["report_to"],
    )

    return training_args


def main():
    # parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    args = parser.parse_args()

    # load cfg
    cfg = _load_cfg(args.config)

    # seed
    _set_seed(cfg["run"]["seed"])

    # build model + tokenizer (LoRA-wrapped model returned)
    model, tokenizer = build_model_and_tokenizer(cfg)

    # data module dispatch (domain-specific)
    data_mod = _load_data_module(cfg)

    # build datasets
    (
        train_dataset,
        eval_dataset,
        tool_eval_examples,
        global_tools,
        max_ctx,  # unused downstream except for possible debug
    ) = data_mod.build_datasets(cfg)

    # collator bound to pad_token_id
    data_collator = make_causal_lm_collate_fn(
        pad_token_id=tokenizer.pad_token_id,
    )

    # training args
    training_args = _build_training_args(cfg)

    # build trainer with custom eval logic
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
    )

    # train and save LoRA adapter
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
