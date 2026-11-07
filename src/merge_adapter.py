"""Влить LoRA-адаптер в веса базовой модели и сохранить обычную модель Hugging Face.

    python src/merge_adapter.py --base_model models/Qwen_Qwen3-4B-Instruct-2507 \
        --lora runs/qwen3-4b-brand/adapter --out merged/qwen3-4b-brand

Влитая модель работает с той же скоростью, что и базовая (адаптер не считается отдельно),
и её можно перевести в GGUF для llama.cpp и Ollama (команды — в README, раздел «Запуск в Ollama»).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.lora).save_pretrained(args.out)
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
