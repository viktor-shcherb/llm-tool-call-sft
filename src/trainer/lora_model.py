from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def _map_dtype(dtype_str: str):
    """
    Map config string to torch dtype.
    Allowed: "bfloat16", "float16", "float32".
    """
    ds = dtype_str.lower()
    if ds == "bfloat16":
        return torch.bfloat16
    if ds in ("float16", "fp16", "half"):
        return torch.float16
    if ds in ("float32", "fp32", "full"):
        return torch.float32
    raise ValueError(f"unsupported torch_dtype '{dtype_str}'")


def _ensure_pad_tokens(tokenizer, model, pad_token_fallback: str):
    """
    Ensure tokenizer.pad_token and model.config.pad_token_id are set.

    pad_token_fallback:
      "eos" means copy eos into pad if missing.
    """
    # tokenizer side
    if tokenizer.pad_token is None:
        if pad_token_fallback == "eos":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("pad_token missing and unsupported pad_token_fallback")

    pad_id = tokenizer.pad_token_id

    # model side
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = pad_id

    return pad_id


def _build_lora_cfg(lora_cfg_dict: Dict[str, Any]) -> LoraConfig:
    """
    Create LoraConfig from cfg['lora'] dict.
    Required keys in cfg['lora']:
      r
      alpha
      dropout
      target_modules
      bias
      task_type
    """
    return LoraConfig(
        r=lora_cfg_dict["r"],
        lora_alpha=lora_cfg_dict["alpha"],
        lora_dropout=lora_cfg_dict["dropout"],
        target_modules=lora_cfg_dict["target_modules"],
        bias=lora_cfg_dict["bias"],
        task_type=lora_cfg_dict["task_type"],
    )


def build_model_and_tokenizer(
    cfg: Dict[str, Any],
) -> Tuple[torch.nn.Module, Any]:
    """
    Create tokenizer, base model, then wrap model with LoRA.

    Expects cfg sections:
      cfg["model"] = {
        "base_model_name": str,
        "torch_dtype": "bfloat16" | "float16" | "float32",
        "pad_token_fallback": "eos"
      }

      cfg["lora"] = {
        "r": int,
        "alpha": int,
        "dropout": float,
        "target_modules": [str,...],
        "bias": "none",
        "task_type": "CAUSAL_LM"
      }
    """
    model_cfg = cfg["model"]
    lora_cfg_dict = cfg["lora"]

    base_model_name = model_cfg["base_model_name"]
    torch_dtype = _map_dtype(model_cfg["torch_dtype"])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch_dtype,
    )

    # pad token handling
    _ensure_pad_tokens(
        tokenizer=tokenizer,
        model=model,
        pad_token_fallback=model_cfg["pad_token_fallback"],
    )

    # build LoRA config and wrap
    lora_cfg = _build_lora_cfg(lora_cfg_dict)
    model = get_peft_model(model, lora_cfg)

    # optional sanity print, mirrors original script
    # caller can silence by redirecting stdout if needed
    model.print_trainable_parameters()

    return model, tokenizer


__all__ = [
    "build_model_and_tokenizer",
]
