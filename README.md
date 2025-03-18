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
