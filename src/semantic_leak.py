"""Поиск обучающих вопросов, похожих по смыслу на отложенные (эмбеддинги bge-m3 через Ollama).

    python src/semantic_leak.py --host http://127.0.0.1:11434 --raw   # все пары от 0,65 и 8 ближайших от 0,55

Буквенная проверка в build_dataset.py ловит только почти дословные повторы. Здесь вопросы сравниваются по смыслу:
для каждого отложенного вопроса — все обучающие с косинусом не ниже --threshold и ещё --top ближайших с косинусом
не ниже --floor (пересказ почти всегда среди ближайших, даже при невысоком косинусе). Кандидаты пишутся
в data/leak_candidates.json; решение «тот же вопрос другими словами или просто та же тема» принимается
отдельно и записывается в data/leak_decisions.json, а build_dataset.py выбрасывает отмеченные пары.
Только стандартная библиотека Python.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def embed(host: str, model: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        body = {"model": model, "input": texts[i:i + 32]}
        req = urllib.request.Request(f"{host}/api/embed", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            out += json.loads(resp.read())["embeddings"]
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--threshold", type=float, default=0.65, help="все пары не ниже этого косинуса")
    ap.add_argument("--top", type=int, default=8, help="плюс столько ближайших на каждый отложенный вопрос...")
    ap.add_argument("--floor", type=float, default=0.55, help="...если их косинус не ниже этого")
    ap.add_argument("--raw", action="store_true", help="сравнивать со всеми партиями data/raw, а не с data/train.json")
    args = ap.parse_args()
    tests = json.loads((ROOT / "data" / "test_questions.json").read_text(encoding="utf-8"))
    if args.raw:
        train = [r for p in sorted((ROOT / "data" / "raw").glob("*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
    else:
        train = json.loads((ROOT / "data" / "train.json").read_text(encoding="utf-8"))
    te = embed(args.host, args.model, [t["question"] for t in tests])
    tr = embed(args.host, args.model, [r["instruction"] for r in train])
    cands = []
    for t, tv in zip(tests, te):
        sims = sorted(((cosine(tv, rv), r) for r, rv in zip(train, tr)), key=lambda x: -x[0])
        for rank, (s, r) in enumerate(sims):
            if s < args.threshold and not (rank < args.top and s >= args.floor):
                break
            cands.append({"test_id": t["id"], "test_question": t["question"], "train_id": r["id"],
                          "train_question": r["instruction"], "cosine": round(s, 3)})
    path = ROOT / "data" / "leak_candidates.json"
    path.write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"отложенных {len(tests)}, обучающих {len(train)}, кандидатов: {len(cands)}, "
          f"отложенных с кандидатами: {len({c['test_id'] for c in cands})} -> {path}")


if __name__ == "__main__":
    main()
