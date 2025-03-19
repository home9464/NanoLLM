* This project aims to train a super-small language model **NanoLLM** with only 3 RMB cost and 2 hours,
  starting completely from scratch.
* The **NanoLLM** series is extremely lightweight, with the smallest version being $\frac{1}{7000}$ the size of GPT-3,
  making it possible to train quickly on even the most ordinary personal GPUs.
* The project also open-sources the minimalist structure of the large model, including extensions for shared mixed
  experts (MoE), dataset cleaning, pretraining, supervised fine-tuning (SFT), LoRA fine-tuning, direct preference
  optimization (DPO) algorithms, and model distillation algorithms, along with the full code of the process.
* All core algorithm code is reconstructed from scratch using native PyTorch! It does not rely on abstract interfaces
  provided by third-party libraries.
* This is not only a full-stage open-source reproduction of a large language model but also a tutorial for beginners in
  LLM.

## 0. download necessary datasets

### 0.1 download dataset to pretrain the foundation model

```bash
source .venv/bin/activate
```

```python
from datasets import load_dataset
dataset_name = "Skylion007/openwebtext"  # or "stas/openwebtext-10k" with 10k lines
ds = load_dataset(dataset_name, trust_remote_code=True)
for split in ds.keys():  # 'train'
    ds[split].to_json(f"{dataset_name.replace('/', '_')}_{split}.jsonl", orient="records", lines=True)
```
The result file **Skylion007_openwebtext_train.jsonl** is about ~38G, 8M lines with following format:

```json
{"text": "a brown fox jumps onto a wolf"},
{"text": "I am a poem"}
```

if it is too big then get a subset from it:

```bash
head -n 1000000 Skylion007_openwebtext_train.jsonl > openwebtext-1M.jsonl
```

### 0.2 download dataset to instruction fine-tune the pretrained model

```bash
curl -OL https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
```

The result file **Skylion007_openwebtext_train.jsonl** is about ~13M, 15k lines with following format:

```json
    {
        "instruction": "When did Virgin Australia start operating?", 
        "context": "Virgin Australia...", 
        "response": "Virgin Australia ...",
        "category": "closed_qa"
    },

    {
      "instruction": "Which is a species of fish? Tope or Rope", 
      "context": "", 
      "response": "Tope", 
      "category": "classification"
    }
```

## 1. build BPE tokenizer
```bash
uv run train_tokenizer.py
```
## 2. pre-train
```bash
uv run train_pretrain.py --use_wandb
```
## 3. instruction fine-tune
```bash
uv run train_full_sft.py --use_wandb
```
## 4. evaluate fine-tuned model
```bash
uv run eval_model.py --model 1
```
