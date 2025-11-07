import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from jinja2 import Template
from openai import OpenAI

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def render_system(sys_tmpl: Template, sysprompt_text: str, tools_json: str) -> str:
    return sys_tmpl.render(sysprompt=sysprompt_text, tools_json=tools_json)


def _process_turn(turn: dict):
    turn_d = {
        "role": turn.get("role"),
        "content": turn.get("content"),
    }
    if turn.get("tool_calls"):
        turn_d["tool_calls"] = turn["tool_calls"]
    return turn_d


def render_user(
    user_tmpl: Template,
    context_obj,
    candidate_a_obj,
    candidate_b_obj,
) -> str:
    context_json = json.dumps({
        "conversation": [
            _process_turn(turn)
            for turn in context_obj["conversation"]
        ],
        "date": context_obj["inferred_date"]
    },  ensure_ascii=False)
    candidate_a_json = json.dumps({
        "content": candidate_a_obj["content"],
        "tool_calls": candidate_a_obj["tool_calls"] or []
    }, ensure_ascii=False)
    candidate_b_json = json.dumps({
        "content": candidate_b_obj["content"],
        "tool_calls": candidate_b_obj["tool_calls"] or []
    }, ensure_ascii=False)
    return user_tmpl.render(
        context_json=context_json,
        candidate_a_json=candidate_a_json,
        candidate_b_json=candidate_b_json,
    )


def load_sysprompt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_tools_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_pair(
    client: OpenAI,
    judge_model: str,
    system_msg: str,
    user_msg: str,
) -> str:
    # Expect exactly "A" / "B" / "tie"
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content.strip()
    # normalize
    text = text.lower()
    if text in ("a", "candidate a"):
        return "A"
    if text in ("b", "candidate b"):
        return "B"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--judge-model", required=True, help="OpenAI model used to judge")

    ap.add_argument("--dataset", required=True, help="HF hub dataset, e.g. org/name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target-column", required=True, help="model column to evaluate")
    ap.add_argument("--push-to", default=None, help="HF hub repo to push to; default = --dataset")

    ap.add_argument("--preference-sysprompt", default="data/preference_sysprompt.jinja2")
    ap.add_argument("--preference-user-template", default="data/preference_user_template.jinja2")
    ap.add_argument("--assist-sysprompt-md", default="data/sysprompt.md")
    ap.add_argument("--tools-json", default="data/tool_calls.json")

    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    # templates and static context
    sys_tmpl = load_template(args.preference_sysprompt)
    user_tmpl = load_template(args.preference_user_template)
    sysprompt_text = load_sysprompt_md(args.assist_sysprompt_md)
    tools_obj = load_tools_json(args.tools_json)
    tools_json_str = json.dumps(tools_obj, ensure_ascii=False)

    system_msg = render_system(sys_tmpl, sysprompt_text, tools_json_str)

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds)

    target_col = args.target_column
    if target_col not in ds.column_names:
        raise ValueError(f"target column {target_col} not found in dataset")

    annotations = [None] * n

    def _process_one(idx: int, row: dict):
        context_obj = row["context"]
        gold_obj = row["gold_response"]
        model_obj = row[target_col]

        if model_obj.get("error") or (not model_obj.get("content") and not model_obj.get("tool_calls")):
            print(f"Skipped {idx}")
            return idx, {
                "content": model_obj.get("content"),
                "tool_calls": model_obj.get("tool_calls", []),
                "preferred": None,
            }

        # keep original model content to write back
        if isinstance(model_obj, str):
            model_obj = json.loads(model_obj)
        if isinstance(gold_obj, str):
            gold_obj = json.loads(gold_obj)

        # randomize order
        if random.random() < 0.5:
            # A = gold, B = model
            order = ("gold", "model")
            candidate_a = gold_obj
            candidate_b = model_obj
        else:
            # A = model, B = gold
            order = ("model", "gold")
            candidate_a = model_obj
            candidate_b = gold_obj

        user_msg = render_user(
            user_tmpl=user_tmpl,
            context_obj=context_obj,
            candidate_a_obj=candidate_a,
            candidate_b_obj=candidate_b,
        )

        try:
            winner = judge_pair(
                client=client,
                judge_model=args.judge_model,
                system_msg=system_msg,
                user_msg=user_msg,
            )
        except Exception as e:
            print(e)
            return idx, {
                "content": model_obj.get("content"),
                "tool_calls": model_obj.get("tool_calls", []),
                "preferred": None,
            }

        # map to +1/0/-1
        if order == ("gold", "model"):
            if winner == "a":
                preferred = -1
            elif winner == "b":
                preferred = 1
            elif winner == "A":
                preferred = -1
            elif winner == "B":
                preferred = 1
            else:
                preferred = 0
        else:  # ("model", "gold")
            if winner == "a" or winner == "A":
                preferred = 1
            elif winner == "b" or winner == "B":
                preferred = -1
            else:
                preferred = 0

        # write back into model_obj
        updated_model_obj = {
            "content": model_obj.get("content"),
            "tool_calls": model_obj.get("tool_calls", []),
            "preferred": preferred,
        }

        return idx, updated_model_obj

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_process_one, i, ds[i]) for i in range(n)]
        for fut in tqdm(as_completed(futs), desc="Annotating", total=n):
            idx, updated = fut.result()
            annotations[idx] = updated

    # overwrite column
    if target_col in ds.column_names:
        ds = ds.remove_columns([target_col])
    ds = ds.add_column(target_col, annotations)

    target_repo = args.push_to if args.push_to else args.dataset
    ds.push_to_hub(target_repo, split=args.split)


if __name__ == "__main__":
    main()
