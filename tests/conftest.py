import pytest
import torch
from torch.utils.data import Dataset
from trainer.eval_metrics import IGNORE_INDEX
from transformers import AutoModelForCausalLM, AutoTokenizer


class ToyTokenizedDataset(Dataset):
    """
    Minimal dataset wrapper for eval_perplexity.
    Each item is a dict with LongTensors:
      input_ids, attention_mask, labels
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        # convert lists -> tensors so eval_perplexity can pad_sequence them
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    """
    Load the tiny instruct model used for smoke eval.
    We set pad_token_id if missing so eval_perplexity does not crash.
    """
    model_name = "viktoroo/gemma-3-270m-tools" # "HuggingFaceTB/SmolLM2-360M-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if model.config.pad_token_id is None:
        # fall back to eos_token_id for padding
        model.config.pad_token_id = model.config.eos_token_id
    return model, tok


@pytest.fixture
def toy_tokenized_dataset():
    """
    Build a tiny dataset of 2 sequences.
    We pretend:
    - tokens [10,11,12,13]
    - assistant tokens are last two tokens
    So labels are [-100,-100,12,13].
    """
    samples = [
        {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "labels": [IGNORE_INDEX, IGNORE_INDEX, 12, 13],
        },
        {
            "input_ids": [20, 21, 22],
            "attention_mask": [1, 1, 1],
            "labels": [IGNORE_INDEX, 21, 22],
        },
    ]
    return ToyTokenizedDataset(samples)


@pytest.fixture
def global_tools_fixture():
    """
    Minimal global tool registry. Matches training format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_inventory",
                "description": "Return availability details for a specific vehicle VIN.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vin": {
                            "type": "string",
                            "description": "Vehicle VIN the shopper asked about."
                        },
                    },
                    "required": ["vin"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_test_drive",
                "description": "Book a test drive appointment for a VIN at a specific datetime.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vin": {
                            "type": "string",
                            "description": "Vehicle VIN to test drive."
                        },
                        "datetime": {
                            "type": "string",
                            "description": "Requested start time in ISO 8601."
                        },
                    },
                    "required": ["vin", "datetime"],
                },
            },
        },
    ]


@pytest.fixture
def raw_sessions_fixture(global_tools_fixture):
    """
    Build two synthetic sessions in the same structure you store for eval.

    We make the system prompt explicitly teach function calling behavior:

    - When the shopper asks about a specific VIN or availability, respond ONLY
      with a <tool_call>{...}</tool_call> block that calls lookup_inventory.
      Do not explain, apologize, or refuse. Do not add extra text.

    - When the shopper is just greeting or doing small talk, answer normally
      in plain text. Do NOT call any tool in that case.

    Session A (tool path):
      user asks about availability of VIN 123
      assistant MUST emit lookup_inventory tool call
      tool returns JSON
      assistant gives natural language answer

    Session B (no tool path):
      user just says "Say hi."
      assistant replies plain text with no tool_calls
    """

    # Compose a deterministic tool doc string for the system message so even a generic instruct model
    # sees the tool API and the required output format.
    tool_doc_lines = [
        "You are CarSalesBot for a dealership.",
        "You have function calling capabilities. You can call these functions:",
    ]
    for t in global_tools_fixture:
        fn = t["function"]["name"]
        desc = t["function"]["description"]
        tool_doc_lines.append(f"- {fn}: {desc}")
    tool_doc_lines.extend([
        "",
        "When the shopper asks about a specific vehicle by VIN or asks if a car is in stock:",
        "1. You MUST respond with EXACTLY one <tool_call>...</tool_call> block.",
        "2. Inside that block output valid JSON:",
        '   {"id":"call_1","type":"function","function":{"name":"lookup_inventory","arguments":"{\\"vin\\": \\"<VIN>\\"}"}}',
        "3. Do NOT add natural language before or after the block.",
        "",
        "After I send you the tool result from lookup_inventory, then respond in natural language.",
        "",
        "When the shopper is just greeting or saying hi:",
        "Respond conversationally as plain text.",
        "Do NOT call any tool in that case.",
    ])
    system_prompt = "\n".join(tool_doc_lines)

    # Session where tool SHOULD be called
    session_a = {
        "session_id": "sess_a",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Shopper: I'm looking at a used Honda Civic. The VIN is 123. "
                    "Follow the policy. Return ONLY the tool call JSON in a <tool_call> block "
                    "so the dealership system can check availability."
                ),
            },
            {
                # expected assistant tool call turn
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_inventory",
                            "arguments": '{"vin": "123"}',
                        },
                    }
                ],
            },
            {
                # tool response turn that the runtime would inject
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"vin":"123","available":true,"price_usd":15999}',
            },
            {
                # final natural language answer turn
                "role": "assistant",
                "content": (
                    "Yes. The Civic with VIN 123 is in stock and listed at $15,999."
                ),
            },
        ],
    }

    # Session where NO tool should be called
    session_b = {
        "session_id": "sess_b",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Shopper: Just say hi to me. Do NOT call any tool. "
                    "You are only greeting me."
                ),
            },
            {
                "role": "assistant",
                "content": "Hello. How can I help you with a vehicle today?",
            },
        ],
    }

    return [session_a, session_b]
