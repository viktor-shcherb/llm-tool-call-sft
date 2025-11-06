import argparse
import importlib

import yaml
from openai import OpenAI

from trainer.eval_metrics import eval_tool_calls

import dotenv
dotenv.load_dotenv()


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="Model name as exposed by the server")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--lora-name", default=None, help="LoRA name registered on the vLLM server, e.g. carsales")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    data_mod = importlib.import_module(cfg["data"]["load_module"])
    (
        _train_ds,
        _eval_ds,
        tool_eval_examples,
        global_tools,
        max_ctx_from_data,
    ) = data_mod.build_datasets(cfg)

    eval_cfg = cfg.get("eval", {})
    max_new_tokens = int(eval_cfg.get("max_new_tokens_eval", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    metrics = eval_tool_calls(
        client=client,
        model=args.model,
        examples=tool_eval_examples,
        global_tools=global_tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        lora_name=args.lora_name,
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
