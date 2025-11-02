from typing import List, Dict, Any, Callable
import torch

from trainer.eval_metrics import IGNORE_INDEX


def _pad_to_length(t: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    """
    Right-pad 1D tensor t to target length with pad_value.
    """
    pad_len = length - t.size(0)
    if pad_len <= 0:
        return t
    return torch.nn.functional.pad(t, (0, pad_len), value=pad_value)


def make_causal_lm_collate_fn(pad_token_id: int) -> Callable:
    """
    Factory. Returns a collate_fn(batch) closure bound to pad_token_id.

    Each batch item must have:
      - input_ids: LongTensor [T_i]
      - attention_mask: LongTensor [T_i]
      - labels: LongTensor [T_i]  (IGNORE_INDEX for masked tokens)

    Output dict:
      input_ids: LongTensor [B, T_max]
      attention_mask: LongTensor [B, T_max]
      labels: LongTensor [B, T_max]
    """
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [b["input_ids"] for b in batch]
        attn_list = [b["attention_mask"] for b in batch]
        labels_list = [b["labels"] for b in batch]

        max_len = max(t.size(0) for t in input_ids_list)

        input_ids = torch.stack(
            [_pad_to_length(t, max_len, pad_token_id) for t in input_ids_list],
            dim=0,
        )
        attention_mask = torch.stack(
            [_pad_to_length(t, max_len, 0) for t in attn_list],
            dim=0,
        )
        labels = torch.stack(
            [_pad_to_length(t, max_len, IGNORE_INDEX) for t in labels_list],
            dim=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


__all__ = [
    "make_causal_lm_collate_fn",
]
