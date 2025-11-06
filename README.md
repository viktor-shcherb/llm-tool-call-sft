# llm-tool-call-sft

This repo fine-tunes a tool-calling chat model for a given domain (car sales, insurance, etc.) using supervised learning with LoRA.
The pipeline prepares conversation data, injects deterministic system prompts and tool schemas, filters long sequences, trains a LoRA adapter, and evaluates both perplexity and tool-call quality.

The design:

* Domain logic lives in its own package (example: `car_sales/`).
* Training and evaluation logic is shared and lives in `trainer/`.
* All run-time knobs (model, LoRA, data, training loop, eval) live in a YAML config.
* `scripts/train.py` is universal and just consumes that YAML.

---

## Repo structure

```text
repo/
├─ configs/
│  └─ car_sales.yaml            # full run config (data, model, lora, train, eval)
│
├─ data/
│  ├─ sysprompt.md              # system prompt template for this domain
│  └─ tool_calls.json           # tool spec (function name, params, description)
│
├─ scripts/
│  └─ train.py                  # entry point for training
│
├─ src/
│  ├─ car_sales/
│  │  ├─ __init__.py            # exposes build_datasets(cfg)
│  │  ├─ prompt.py             # system prompt assembly, timestamp logic
│  │  └─ data_prep.py           # dataset mapping, tokenization, filtering
│  │
│  └─ trainer/
│     ├─ __init__.py
│     ├─ lora_model.py          # load base model + tokenizer, inject LoRA
│     ├─ collate.py             # batch collation with padding
│     ├─ eval_metrics.py        # perplexity + tool-call eval metrics
│     └─ tool_trainer.py        # HF Trainer subclass that logs custom eval
│
├─ tests/
│  ├─ test_prompts.py
│  ├─ test_tokenize_with_role_spans.py
│  ├─ test_build_datasets.py
│  └─ test_collate.py
│
├─ pyproject.toml               # marks `src/` as the package root
├─ requirements.txt             # frozen working environment
└─ README.md                    # this file
```

High-level flow:

1. The dataset is loaded and converted into chat-style messages with per-turn tool calls.
2. Each message sequence is tokenized with a chat template that includes a shuffled tool registry.
3. Only assistant tokens are labeled for loss. User/system/tool messages are masked to `-100`.
4. Sequences longer than the model context are dropped.
5. A base causal LM is loaded and wrapped with LoRA adapters.
6. Training runs with Hugging Face `Trainer`, but evaluation is extended:

   * Perplexity on eval split.
   * Tool-call precision/recall/F1, argument correctness, hallucination rate.

---

## Config

Every run is driven by a YAML file in `configs/`. Example key blocks:

```yaml
data:
  load_module: "car_sales"
  dataset_name: "Salesteq/car-sales-convos"
  timezone: "Europe/Zurich"
  sysprompt_path: "data/sysprompt.md"
  tools_path: "data/tool_calls.json"

  tokenizer: "viktoroo/SmolLM2-360M-Tools"
  max_context_length: 8192
  drop_oversized: true
  n_tool_sessions_eval: 64

model:
  base_model_name: "viktoroo/SmolLM2-360M-Tools"
  torch_dtype: "bfloat16"
  pad_token_fallback: "eos"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"
  task_type: "CAUSAL_LM"

train:
  # basic
  output_dir: "checkpoints/car-sales-sft"
  seed: 42

  # logging
  report_to: "wandb"

  # core trainer args
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  num_train_epochs: 3

  learning_rate: 0.0002
  weight_decay: 0.0
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1

  bf16: true
  fp16: false

  ddp_find_unused_parameters: false

  logging_steps: 10
  eval_strategy: "epoch"
  save_strategy: "epoch"
  save_total_limit: 2
  overwrite_output_dir: false
  remove_unused_columns: false

  # hub
  push_to_hub: true
  hub_model_id: "Salesteq/SmolLM2-360M-CarSales"
  hub_private_repo: true

eval:
  max_new_tokens_eval: 256
  temperature: 0.0
```

What each section controls:

* `data`: dataset-specific choices. Where to load the HF dataset from, which domain code to use, how long sequences are allowed to be, which tokenizer to use for tokenization, how many eval sessions to sample for tool-call metrics.
* `model`: base LM load settings. Which pretrained model to start from, precision, and how to set pad token if missing.
* `lora`: LoRA adapter config. Rank, alpha, dropout, and which module names to inject adapters into.
* `train`: training loop hyperparameters. These map directly to `transformers.TrainingArguments` (batch sizes, lr, scheduler, epochs, logging, save strategy, precision flags).
* `eval`: inference settings for evaluation. How many tokens to generate when simulating tool calls and what decoding temperature to use.

---

## How training works

1. `scripts/train.py`:

   * loads the YAML config
   * sets the random seed
   * dynamically imports `cfg["data"]["load_module"]`
   * calls `build_datasets(cfg)` from that module

     * that returns: train_dataset, eval_dataset, tool_eval_examples, global_tools, max_ctx
   * builds the base model + tokenizer and applies LoRA (`trainer/lora_model.py`)
   * builds a collator that pads batches (`trainer/collate.py`)
   * creates `TrainingArguments` from `cfg["train"]` and `cfg["run"]`
   * instantiates `ToolTrainer` with:

     * datasets
     * model
     * tokenizer (as processing class)
     * eval extras (tool_eval_examples, global_tools)
   * calls `.train()` then `.save_model()`

2. `car_sales/build_datasets(cfg)`:

   * loads the raw dataset (`cfg.data.dataset_name`)
   * loads the system prompt template (`data/sysprompt.md`) and tool schemas (`data/tool_calls.json`)
   * builds messages per session:

     * injects a deterministic "Current DateTime (Europe/Zurich): ..." line using the session ID
     * shuffles the system prompt sections and bullet order deterministically per session ID
     * renders each conversation to a chat format that includes tool_calls
   * tokenizes incrementally with `tokenizer.apply_chat_template(...)` so we know which tokens came from which role
   * masks loss for non-assistant tokens by setting label = -100 (`IGNORE_INDEX`)
   * drops any example whose tokenized length exceeds `cfg.data.max_context_length` if `drop_oversized: true`
   * builds `tool_eval_examples` for evaluation (prompts where the assistant should or should not call tools)

3. `trainer/tool_trainer.ToolTrainer` extends Hugging Face `Trainer`:

   * During evaluation it logs:

     * perplexity over eval_dataset (masked loss only on assistant spans)
     * tool-call quality:

       * precision / recall / F1 on function names
       * argument parse success and exact match
       * hallucination rate when the gold data says "no tool call expected"

---

## Restrictions on the base model

The model must follow these constraints:

1. It must be a causal LM

   * Loadable via `transformers.AutoModelForCausalLM.from_pretrained`.

2. It must support the chat template used here

   * Its tokenizer must have `apply_chat_template(...)` and accept
     `tools=...`, `add_generation_prompt=...`,
     `chat_template_kwargs={"enable_thinking": False}`.
   * The model is expected to emit tool calls as `<tool_call>{...json...}</tool_call>`.
     Eval parses that.

3. It must allow adding a pad token

   * If the tokenizer has no `pad_token`, we set `tokenizer.pad_token = tokenizer.eos_token`.
   * We then set `model.config.pad_token_id` to match.

4. It must handle the configured context length

   * We assume it can run sequences up to `cfg.data.max_context_length`.
   * Samples longer than that get dropped so we never feed longer sequences than that during training.

5. It must be LoRA-compatible

   * `lora.target_modules` in the config must match real module names in the model (like `q_proj`, `v_proj`, `gate_proj`, etc.).
   * If your backbone uses different names, update the YAML.

---

## How to add a new dataset / domain

You do not touch `trainer/` or `scripts/train.py`.

1. Copy `src/car_sales/` to `src/<new_domain>/`.
2. Edit `<new_domain>/prompts.py` and `<new_domain>/data_prep.py`:

   * Change how conversations are turned into messages (`session_to_messages`)
   * Change which system prompt file and tool spec file you want
   * Change timezone if needed
   * Change dataset name
3. In `<new_domain>/__init__.py` expose `build_datasets(cfg)`.
4. Create a new config `configs/<new_domain>.yaml` with:

   ```yaml
   data:
     load_module: "<new_domain>"
     dataset_name: "<hf_dataset_name>"
     ...
   ```
5. Train:

   ```bash
   python scripts/train.py --config configs/<new_domain>.yaml
   ```

No other code changes required because `scripts/train.py` dynamically imports `cfg.data.load_module`.

---

## Setup on a fresh machine

You want to recreate your working virtual environment exactly.

1. Clone the repo.
   
2. Install all dependencies from `pyproject.toml`:
1) For training
```bash
uv pip install .
```
2) For serving and evaluation
```bash
uv pip install .[serve]
```   

3. Run tests:

   ```bash
   uv run pytest -q
   ```

4. Run training:

   ```bash
   uv run python scripts/train.py --config configs/car_sales/qwen3-4b.yaml
   ```

## Run evaluation and inference

1. Start the model server (base + LoRA)

   ```bash
   uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
     --trust-remote-code \
     --enable-lora \
     --lora-modules carsales=Salesteq/Qwen3-4B-Instruct-2507-CarSales \
     --host 0.0.0.0 \
     --port 8000 \
     --max_model_len=8192 \
     --enable-auto-tool-choice \
     --tool-call-parser xlam
   ```
   This exposes an OpenAI-compatible endpoint at `http://0.0.0.0:8000/v1` and registers the LoRA under the name `carsales`.

2. Run the evaluation script against that server

   ```bash
   PYTHONPATH=src uv run python scripts/eval.py \
     --config configs/car_sales/qwen3-4b.yaml \
     --endpoint http://127.0.0.1:8000/v1 \
     --model Qwen/Qwen3-4B-Instruct-2507 \
     --lora-name carsales
   ```
   The script:

   * loads the dataset from `configs/car_sales.yaml`
   * sends tool-eval turns to the server
   * passes `extra_body={"lora_name": "carsales"}` so the adapter is used
   * prints the tool-call metrics

3. Do ad-hoc inference (Python)

   ```bash
   uv run python - << 'PY'

    from openai import OpenAI
    client = OpenAI(base_url="[http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)", api_key="EMPTY")
    
    resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[{"role": "user", "content": "Write a follow-up to a car lead about the Audi A4."}],
    extra_body={"lora_name": "carsales"},
    )
    print(resp.choices[0].message.content)
    PY

    ```

4. Do ad-hoc inference (curl)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Give me a short reply to a car prospect."}],
    "extra_body": {"lora_name": "carsales"}
  }'
```

---

5. Measure throughput and latency (vLLM bench)

   vLLM ships a simple load tool. Run it against the same model to see tokens/s and latency:

   ```bash
   uv run vllm bench serve \
      --backend openai \                                                                              
      --base-url http://127.0.0.1:8000 \
      --dataset-name random \
      --lora-modules carsales \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --max-concurrency 64
   ```

vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --dataset-name random --model google/gemma-3-4b-it --max-concurrency 64

After training finishes, the LoRA adapter and metrics are written under `run.output_dir` from the config.


