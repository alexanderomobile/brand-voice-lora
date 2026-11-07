"""Ответы на отложенные вопросы через локальную Ollama (модель после GGUF и квантования).

    python src/ollama_answers.py --host http://127.0.0.1:11434 --model brand-voice:q4_k_m \
        --out eval/answers_ollama-q4.json

Режимы: ollama — без системного промпта, ollama_facts — со справкой style/facts_prompt.txt.
Только стандартная библиотека Python: так же модель можно звать из любого проекта, например из бота.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def chat(host: str, model: str, question: str, system: str | None) -> dict[str, Any]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_predict": 400, "repeat_penalty": 1.1}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "test_questions.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    facts = (ROOT / "style" / "facts_prompt.txt").read_text(encoding="utf-8").strip()
    rows = []
    for mode, system in (("ollama", None), ("ollama_facts", facts)):
        for q in questions:
            t0 = time.time()
            r = chat(args.host, args.model, q["question"], system)
            gen_s = r.get("eval_duration", 0) / 1e9
            rows.append({"id": q["id"], "category": q.get("category"), "mode": mode, "question": q["question"],
                         "answer": r["message"]["content"].strip(), "prompt_tokens": r.get("prompt_eval_count", 0),
                         "new_tokens": r.get("eval_count", 0), "seconds": round(gen_s or time.time() - t0, 3),
                         "wall_seconds": round(time.time() - t0, 3)})
            print(f"{mode:13s} {q['id']}: {r.get('eval_count')} ток., {gen_s:.1f} с", flush=True)
    meta = {"base_model": args.model, "host": args.host, "decoding": "ollama temperature 0, repeat_penalty 1.1",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "answers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
