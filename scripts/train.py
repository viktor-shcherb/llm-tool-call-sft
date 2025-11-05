import argparse
import importlib
from typing import Any, Dict

import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainingArguments, Trainer

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


def run_vllm_tool_eval_and_log(
    trainer: Trainer,
    tokenizer,
    tool_eval_examples,
    global_tools,
    max_ctx: int,
    prefix: str,
    fsdp_wrapped: bool,
):
    # 1) write model+tokenizer to disk on rank 0
    if is_main_process():
        if fsdp_wrapped:
            # after Trainer wrapped the model (after .train()), we can use the HF helper
            trainer.save_model()
        else:
            # before training: model is the plain nn.Module, so save it directly
            trainer.model.save_pretrained(trainer.args.output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(trainer.args.output_dir)
    barrier()

    # 2) only rank 0 runs vLLM
    if not is_main_process():
        barrier()
        return

    metrics = eval_tool_calls(
        model_path=trainer.args.output_dir,
        tokenizer=tokenizer,
        examples=tool_eval_examples,
        global_tools=global_tools,
        max_model_len=max_ctx,
        max_new_tokens=256,
        temperature=0.0,
        tensor_parallel_size=1,
    )

    trainer.log_metrics(prefix, metrics)
    trainer.save_metrics(prefix, metrics)
    barrier()


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
        tokenizer_for_tools=tokenizer,
    )

    # pre-train vLLM tool eval (model not FSDP-wrapped yet)
    run_vllm_tool_eval_and_log(
        trainer=trainer,
        tokenizer=tokenizer,
        tool_eval_examples=tool_eval_examples,
        global_tools=global_tools,
        max_ctx=max_ctx,
        prefix="tool_eval_pre",
        fsdp_wrapped=False,
    )

    trainer.train()

    # post-train vLLM tool eval (now Trainer has done the wrapping)
    run_vllm_tool_eval_and_log(
        trainer=trainer,
        tokenizer=tokenizer,
        tool_eval_examples=tool_eval_examples,
        global_tools=global_tools,
        max_ctx=max_ctx,
        prefix="tool_eval_post",
        fsdp_wrapped=True,
    )

    if is_main_process():
        trainer.push_to_hub(commit_message="train: finish")


if __name__ == "__main__":
    main()
