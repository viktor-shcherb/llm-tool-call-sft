import torch
from car_sales.data_prep import tokenize_with_role_spans, IGNORE_INDEX

class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        chat_template_kwargs=None,
        return_tensors=None,
    ):
        # pretend each message adds 10 tokens
        n = len(messages)
        # prefix length for n messages is n * 10 tokens [0..n*10-1]
        return list(range(n * 10))

def test_tokenize_with_role_spans_masking_and_shapes():
    tokenizer = FakeTokenizer()
    global_tools = [
        {"type": "function", "function": {"name": "lookup"}},
        {"type": "function", "function": {"name": "quote"}},
    ]

    # 4 turns: system, user, assistant, tool
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "ok", "tool_call_id": "abc"},
    ]

    out = tokenize_with_role_spans(
        messages=messages,
        session_id="sess-1",
        tokenizer=tokenizer,
        global_tools=global_tools,
    )

    input_ids = out["input_ids"]
    labels = out["labels"]
    attn = out["attention_mask"]

    # expected lengths: 4 messages * 10 tokens each = 40 tokens total
    assert input_ids.shape == torch.Size([40])
    assert labels.shape == torch.Size([40])
    assert attn.shape == torch.Size([40])

    # tokens per message:
    # msg0:   idx [0..9]
    # msg1:   idx [10..19]
    # msg2:   idx [20..29]  <-- assistant, should be supervised
    # msg3:   idx [30..39]
    #
    # Check labels mask for first user turn range [0..19] -> all IGNORE_INDEX
    assert torch.all(labels[0:20] == IGNORE_INDEX)

    # Assistant range [20..29] should equal the same token ids and not IGNORE_INDEX
    assert torch.equal(labels[20:30], input_ids[20:30])
    assert torch.all(labels[20:30] != IGNORE_INDEX)

    # After assistant, tool message [30..39] should be IGNORE_INDEX again
    assert torch.all(labels[30:40] == IGNORE_INDEX)

    # attention_mask should be 1 everywhere
    assert torch.all(attn == 1)

def test_tokenize_with_role_spans_is_deterministic_over_tool_shuffle():
    class FakeTokenizer2(FakeTokenizer):
        # identical behavior, we only vary session_id to test determinism later
        pass

    tokenizer = FakeTokenizer2()

    # global_tools intentionally in a certain order
    tools1 = [
        {"type": "function", "function": {"name": "lookup"}},
        {"type": "function", "function": {"name": "quote"}},
    ]
    # same tools but swapped order to simulate caller passing differently
    tools2 = list(reversed(tools1))

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    out_a = tokenize_with_role_spans(
        messages=messages,
        session_id="same-session",
        tokenizer=tokenizer,
        global_tools=tools1,
    )
    out_b = tokenize_with_role_spans(
        messages=messages,
        session_id="same-session",
        tokenizer=tokenizer,
        global_tools=tools2,
    )

    # same session_id implies deterministic shuffle of tools
    # so output tokenization and labels must match
    assert torch.equal(out_a["input_ids"], out_b["input_ids"])
    assert torch.equal(out_a["labels"], out_b["labels"])
    assert torch.equal(out_a["attention_mask"], out_b["attention_mask"])
