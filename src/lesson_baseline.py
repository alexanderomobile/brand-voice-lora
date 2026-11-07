"""Ответы модели, обученной кодом урока как есть, через его же inference/chat.py.

Функции загрузки и генерации берутся из lesson-code без изменений; промпт — как в chat_loop урока
для первого сообщения: "User: <вопрос>\nAssistant:". Генерация урока со случайной выборкой
(temperature 0.7), поэтому зерно фиксировано.

    python src/lesson_baseline.py --lesson_dir ../lesson-code --base_model ai-forever/rugpt3small_based_on_gpt2 \
        --lora ../lesson-code/fine_tuning/lora_brand --out eval/answers_lesson_rugpt3small.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lesson_dir", type=Path, required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "test_questions.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # inference/chat.py урока грузится явно по пути: у этого проекта есть свой src/chat.py с тем же именем
    spec = importlib.util.spec_from_file_location("lesson_chat", args.lesson_dir / "inference" / "chat.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"нет {args.lesson_dir / 'inference' / 'chat.py'}")
    lesson_chat: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson_chat)

    model, tokenizer = lesson_chat.load_model_and_tokenizer(args.base_model, str(args.lora))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        torch.manual_seed(args.seed)
        prompt = f"User: {q['question']}\nAssistant:"
        prompt_tokens = len(tokenizer.encode(prompt))
        t0 = time.time()
        answer = lesson_chat.generate_response(model, tokenizer, prompt)  # параметры по умолчанию из урока
        seconds = time.time() - t0
        new_tokens = len(tokenizer.encode(answer)) if answer else 0
        rows.append({"id": q["id"], "category": q.get("category"), "mode": "lesson", "question": q["question"],
                     "answer": answer, "prompt_tokens": prompt_tokens, "new_tokens": new_tokens,
                     "seconds": round(seconds, 3)})
        print(f"lesson {q['id']}: {new_tokens} ток., {seconds:.1f} с", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    meta = {"base_model": args.base_model, "lora": str(args.lora), "gpu": gpu,
            "decoding": "lesson chat.py: sampling t=0.7 top_p=0.9 top_k=50 rp=1.1, max_length=512, seed fixed",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "answers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
