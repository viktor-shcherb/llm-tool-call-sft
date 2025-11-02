import torch
from trainer.collate import make_causal_lm_collate_fn, IGNORE_INDEX

def test_make_causal_lm_collate_fn_padding():
    pad_token_id = 42
    collate = make_causal_lm_collate_fn(pad_token_id)

    batch = [
        {
            "input_ids": torch.tensor([10, 11, 12], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX, 13, 14], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([20, 21], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([20, IGNORE_INDEX], dtype=torch.long),
        },
    ]

    out = collate(batch)

    # shapes
    assert out["input_ids"].shape == torch.Size([2, 3])
    assert out["attention_mask"].shape == torch.Size([2, 3])
    assert out["labels"].shape == torch.Size([2, 3])

    # last position of second sample should be padded
    assert out["input_ids"][1, 2].item() == pad_token_id
    assert out["attention_mask"][1, 2].item() == 0
    assert out["labels"][1, 2].item() == IGNORE_INDEX

    # first row unchanged
    assert torch.equal(out["input_ids"][0], torch.tensor([10, 11, 12]))
    assert torch.equal(out["attention_mask"][0], torch.tensor([1, 1, 1]))
    assert torch.equal(out["labels"][0], torch.tensor([IGNORE_INDEX, 13, 14]))
