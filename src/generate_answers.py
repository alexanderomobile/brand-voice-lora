"""Ответы на отложенные вопросы в нескольких режимах для сравнения:

- base        — базовая модель как есть, без системного промпта;
- base_prompt — базовая модель, в системном промпте правила стиля и справка с фактами
                (style/style_rules.txt + style/facts_prompt.txt): так стиль задают без дообучения;
- lora        — дообученная модель без системного промпта;
- lora_facts  — дообученная модель, в системном промпте только та же справка с фактами (как в боте с RAG:
                стиль даёт дообучение, факты — найденный текст).

    python src/generate_answers.py --base_model models/Qwen_Qwen3-4B-Instruct-2507 \
        --lora runs/qwen3-4b-brand/adapter --out eval/answers_qwen3-4b.json

Генерация жадная (temperature 0): одинаковый вход — одинаковый ответ, сравнение воспроизводимо. Предел длины —
1200 новых токенов; ответ, упёршийся в него, помечается truncated. Параметры записываются в meta.modes по режимам.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import generate, load  # noqa: E402
from prompting import prompt_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "style"


def system_for(mode: str) -> str | None:
    facts = (STYLE / "facts_prompt.txt").read_text(encoding="utf-8").strip()
    rules = (STYLE / "style_rules.txt").read_text(encoding="utf-8").strip()
    return {"base": None, "lora": None, "base_prompt": f"{rules}\n\n{facts}", "lora_facts": facts}[mode]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "test_questions.json")
    ap.add_argument("--modes", default="base,base_prompt,lora,lora_facts")
    ap.add_argument("--max_new_tokens", type=int, default=1200, help="предел длины ответа; ответ, упёршийся в него, "
                    "помечается truncated")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", action="store_true", help="влить адаптер в веса (быстрее; режимы base* недоступны)")
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    modes = args.modes.split(",")
    model, tokenizer = load(args.base_model, args.lora)
    if args.merge:
        if any(m.startswith("base") for m in modes):
            raise SystemExit("--merge несовместим с режимами base*")
        model = model.merge_and_unload()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    old: dict[str, Any] = (json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists()
                           else {"meta": {}, "answers": []})
    rows = [r for r in old["answers"] if r["mode"] not in modes]  # другие режимы из прошлого прогона сохраняются
    # параметры у каждого режима свои: файл может собираться из нескольких запусков
    mode_meta = {m: v for m, v in old.get("meta", {}).get("modes", {}).items() if m not in modes}
    for mode in modes:
        ctx = model.disable_adapter() if mode.startswith("base") else contextlib.nullcontext()
        sys_text = system_for(mode)
        t_mode = time.time()
        with ctx:
            for q in questions:
                prompt = prompt_text(tokenizer, q["question"], sys_text)
                prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                answer, new_tokens, seconds = generate(model, tokenizer, prompt, args.max_new_tokens)
                rows.append({"id": q["id"], "category": q.get("category"), "mode": mode, "question": q["question"],
                             "answer": answer, "prompt_tokens": prompt_tokens, "new_tokens": new_tokens,
                             "seconds": round(seconds, 3), "merged": bool(args.merge),
                             "truncated": new_tokens >= args.max_new_tokens})
                print(f"{mode:12s} {q['id']}: {new_tokens} ток., {seconds:.1f} с", flush=True)
        print(f"== {mode}: {time.time() - t_mode:.0f} с", flush=True)
        mode_meta[mode] = {
            "decoding": "greedy, repetition_penalty 1.1", "max_new_tokens": args.max_new_tokens,
            "merged": bool(args.merge), "adapter": not mode.startswith("base"),
            "system": {"base": None, "lora": None, "base_prompt": "style_rules.txt + facts_prompt.txt",
                       "lora_facts": "facts_prompt.txt"}[mode],
            "system_sha256": hashlib.sha256(sys_text.encode("utf-8")).hexdigest() if sys_text else None,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    meta = {"base_model": args.base_model, "lora": args.lora, "gpu": gpu, "modes": mode_meta}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "answers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: {args.out} ({len(rows)} ответов)")


if __name__ == "__main__":
    main()
