import random
from typing import Any, Dict, List, Optional

import torch
from transformers import Trainer

from trainer.eval_metrics import (
    eval_perplexity,
    eval_tool_calls,
)


class ToolTrainer(Trainer):
    """
    Extension of Hugging Face Trainer.

    Adds:
    - Perplexity computed over eval_dataset with masked labels.
    - Tool-call quality metrics:
        precision / recall / F1 on function names
        argument parse / match rates
        hallucination rate when no tool should be called

    All custom eval metrics are computed only on rank 0.
    """

    def __init__(
        self,
        *args,
        tokenizer_for_tools,
        tool_eval_examples,
        global_tools,
        max_new_tokens_eval: int,
        temperature_eval: float,
        n_short_eval_examples: int,
        **kwargs,
    ):
        """
        Extra args:
          tokenizer_for_tools: tokenizer used for generation/parsing tool calls
          tool_eval_examples: list[ToolEvalTurn] from prepare_tool_eval_examples
          global_tools: list of tool specs
          max_new_tokens_eval: cap for eval generation
          temperature_eval: decode temperature for eval generation
        """
        super().__init__(*args, **kwargs)

        self.tokenizer_for_tools = tokenizer_for_tools
        self.tool_eval_examples = tool_eval_examples or []
        self.global_tools = global_tools or []
        self.max_new_tokens_eval = max_new_tokens_eval
        self.temperature_eval = temperature_eval
        self.n_short_eval_examples = n_short_eval_examples

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Runs:
          - eval_perplexity on eval_dataset
          - eval_tool_calls on tool_eval_examples
        Aggregates and logs all metrics.
        """
        metrics = {}

        # only main process computes heavy custom eval
        if self.is_world_process_zero():
            eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset

            # 1. Perplexity with label masking taken into account
            ppl_stats = eval_perplexity(
                model=self.model,
                dataset=eval_ds,
                batch_size=self.args.per_device_eval_batch_size,
                device=self.model.device
                if isinstance(self.model, torch.nn.Module)
                else torch.device("cpu"),
            )

            metrics[f"{metric_key_prefix}_ppl"] = ppl_stats["perplexity"]
            metrics[f"{metric_key_prefix}_loss_masked"] = ppl_stats["loss"]
            metrics[f"{metric_key_prefix}_num_tokens"] = ppl_stats["num_tokens"]

            examples = self.tool_eval_examples
            if not metric_key_prefix.startswith("full"):
                examples = random.sample(examples, min(self.n_short_eval_examples, len(examples)))

            # 2. Tool-call behavior quality
            tool_stats = eval_tool_calls(
                model=self.model,
                tokenizer=self.tokenizer_for_tools,
                examples=examples,
                global_tools=self.global_tools,
                device=self.model.device,
                max_new_tokens=self.max_new_tokens_eval,
                temperature=self.temperature_eval,
            )

            for k, v in tool_stats.items():
                metrics[f"{metric_key_prefix}_{k}"] = v

            # log to Trainer's logger and persist to disk under output_dir
            self.log(metrics)
            self.save_metrics(split=metric_key_prefix, metrics=metrics)

        return metrics


__all__ = [
    "ToolTrainer",
]
