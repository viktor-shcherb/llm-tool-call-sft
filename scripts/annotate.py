import argparse
import json
import hashlib
import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from datasets import load_dataset
from openai import OpenAI

import dotenv
from tqdm import tqdm

dotenv.load_dotenv()


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_tools(cfg, config_path: str):
    data_cfg = cfg.get("data", {})
    tools_path = data_cfg.get("tools_path")
    if not tools_path:
        raise ValueError("config.data.tools_path is required")

    cfg_dir = Path(config_path).parent
    if not Path(tools_path).is_absolute() and (cfg_dir / tools_path).is_file():
        tools_file = cfg_dir / tools_path
    else:
        tools_file = Path(tools_path)

    with open(tools_file, "r") as f:
        tools = json.load(f)

    if not isinstance(tools, list):
        raise ValueError("tools file must contain a JSON list")
    return tools


def load_sysprompt(cfg, config_path: str):
    data_cfg = cfg.get("data", {})
    sysprompt_path = data_cfg.get("sysprompt_path")
    if not sysprompt_path:
        raise ValueError("config.data.sysprompt_path is required")

    cfg_dir = Path(config_path).parent
    if not Path(sysprompt_path).is_absolute() and (cfg_dir / sysprompt_path).is_file():
        sp_file = cfg_dir / sysprompt_path
    else:
        sp_file = Path(sysprompt_path)

    with open(sp_file, "r") as f:
        return f.read()


def _deterministic_shuffle_tools(tools, session_id: str):
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    rng = random.Random(seed_int)
    tools_copy = deepcopy(tools)
    rng.shuffle(tools_copy)
    return tools_copy


def _openaiify_messages(messages):
    out = []
    for m in messages:
        m = deepcopy(m)
        tcalls = m.get("tool_calls")
        if tcalls:
            fixed = []
            for tc in tcalls:
                tc = deepcopy(tc)
                tc.setdefault("type", "function")
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    fn["arguments"] = json.dumps(args)
                tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
            m.setdefault("content", "")
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-column", required=True)
    ap.add_argument("--output-column", required=True)
    ap.add_argument("--push-to", default=None)

    ap.add_argument("--lora-name", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    tools = load_tools(cfg, args.config)
    sysprompt_text = load_sysprompt(cfg, args.config)
    timezone_str = cfg["data"]["timezone"]

    # import the prompt submodule to get session_to_messages
    prompt_mod = __import__(f"{cfg['data']['load_module']}.prompt", fromlist=["session_to_messages"])
    session_to_messages = getattr(prompt_mod, "session_to_messages")

    eval_cfg = cfg.get("eval", {})
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(eval_cfg.get("max_new_tokens_eval", 256))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(eval_cfg.get("temperature", 0.0))
    )

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds)
    annotations = [None] * n

    def _annotate_one(idx, row):
        raw = row[args.input_column]
        if isinstance(raw, str):
            example = json.loads(raw)
        else:
            example = raw

        # build messages same as dataset builder
        messages = session_to_messages(
            example=example,
            base_sysprompt_text=sysprompt_text,
            timezone_str=timezone_str,
        )
        messages = _openaiify_messages(messages)

        session_id = example.get("session_id")
        if not session_id:
            session_id = hashlib.sha256(
                json.dumps(example, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]

        shuffled_tools = _deterministic_shuffle_tools(tools, session_id)
        use_model = args.lora_name if args.lora_name else args.model

        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                tools=shuffled_tools,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            out = json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception as e:
            out = json.dumps({"error": str(e)}, ensure_ascii=False)

        return idx, out

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_annotate_one, i, ds[i]) for i in range(n)]
        for fut in tqdm(as_completed(futures), desc="Annotating", total=n):
            idx, out = fut.result()
            annotations[idx] = out

    if args.output_column in ds.column_names:
        ds = ds.remove_columns([args.output_column])

    annotated_ds = ds.add_column(args.output_column, annotations)

    target_repo = args.push_to if args.push_to is not None else args.dataset
    annotated_ds.push_to_hub(target_repo, split=args.split)


if __name__ == "__main__":
    main()
