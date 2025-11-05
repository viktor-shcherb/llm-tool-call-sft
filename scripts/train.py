import argparse
import importlib
import os
from typing import Any, Dict

import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

from trainer.lora_model import build_model_and_tokenizer
from trainer.collate import make_causal_lm_collate_fn
from trainer.eval_metrics import eval_tool_calls

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


def _get_base_model_name(cfg: Dict[str, Any]) -> str:
    # adapt to your actual config keys
    model_cfg = cfg.get("model", {})
    return model_cfg.get("base_model_name")


def _export_merged_for_vllm(
    adapter_dir: str,
    tokenizer,
    cfg: Dict[str, Any],
    export_dir: str,
):
    """
    Build a fresh base model, load LoRA adapter from adapter_dir, merge, and save to export_dir.
    This gives vLLM a complete HF model folder.
    """
    os.makedirs(export_dir, exist_ok=True)

    base_model_name = _get_base_model_name(cfg)
    # load base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # load adapter onto it
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(export_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(export_dir)


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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    if is_main_process():
        trainer.push_to_hub(commit_message="train: finish")


if __name__ == "__main__":
    main()
