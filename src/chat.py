"""Чат с моделью в терминале (по мотивам inference/chat.py из урока).

    python src/chat.py --base_model models/Qwen_Qwen3-4B-Instruct-2507 --lora runs/qwen3-4b-brand/adapter
    python src/chat.py --base_model ... --lora ... --ask "Что такое RAG?"      # один вопрос без диалога

Команды в чате: quit / exit / q — выход, clear — очистить историю.
Модель работает локально: вопросы никуда не отправляются.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompting import prompt_text  # noqa: E402


def load(base_model: str, lora: str | None) -> tuple[Any, Any]:
    use_gpu = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_gpu else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(lora or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to("cuda" if use_gpu else "cpu")
    if lora:
        model = PeftModel.from_pretrained(model, lora)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 400, temperature: float = 0.0,
             repetition_penalty: float = 1.1) -> tuple[str, int, float]:
    """(ответ, сколько токенов сгенерировано, секунды). temperature 0 — жадная генерация, воспроизводимо."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         **sampling)
    seconds = time.time() - t0
    new = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True).strip()
    if "### Instruction:" in text:  # модель без chat template может начать следующий вопрос сама
        text = text.split("### Instruction:")[0].strip()
    return text, int(new.shape[0]), seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora", help="папка адаптера (без неё — базовая модель)")
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--ask", help="задать один вопрос и выйти")
    args = ap.parse_args()

    model, tokenizer = load(args.base_model, args.lora)
    system = args.system_prompt or None
    if args.ask:
        answer, n, s = generate(model, tokenizer, prompt_text(tokenizer, args.ask, system),
                                args.max_new_tokens, args.temperature)
        print(answer)
        print(f"[{n} токенов за {s:.1f} с]", file=sys.stderr)
        return

    print("Чат запущен. quit — выход, clear — очистить историю.\n")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "clear":
            history = []
            print("История очищена.\n")
            continue
        answer, n, s = generate(model, tokenizer, prompt_text(tokenizer, question, system, history[-5:]),
                                args.max_new_tokens, args.temperature)
        print(f"\nПомощник: {answer}\n[{n} токенов, {s:.1f} с]\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
