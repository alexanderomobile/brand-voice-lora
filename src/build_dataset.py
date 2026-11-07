"""Сборка обучающего датасета из партий data/raw/*.json.

    python src/build_dataset.py

Что делает: проверяет поля, убирает повторы вопросов внутри датасета и вопросы, похожие на отложенные
(data/test_questions.json), прогоняет линтер стиля и пишет data/train.json и data/dataset_report.json.
Похожесть — доля общих буквенных 4-грамм (Жаккар): 0,6 и выше внутри датасета — повтор, 0,5 и выше с отложенным
вопросом целиком или с любым его предложением — утечка (составной вопрос иначе прячет совпадение одной части).
Буквы ловят только почти дословные совпадения. Тот же вопрос другими словами ищет src/semantic_leak.py
(эмбеддинги), решения по кандидатам — data/leak_decisions.json: пара, которую хотя бы один разметчик признал
тем же вопросом, в обучение не идёт.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_lint import lint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DUP_THRESHOLD = 0.6
LEAK_THRESHOLD = 0.5


def grams(text: str, n: int = 4) -> set[str]:
    t = " ".join(re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е")))
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def load_leak_decisions(path: Path) -> dict[str, str]:
    """train_id -> test_id для пар, признанных тем же вопросом (хотя бы одним разметчиком)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for d in json.loads(path.read_text(encoding="utf-8")):
        if d.get("same"):
            out.setdefault(d["train_id"], d["test_id"])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def build(raw_dir: Path, test_path: Path,
          decisions_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows.append(row)
    report: dict[str, Any] = {"raw": len(rows), "dropped": []}
    tests = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else []
    test_grams = [(t["id"], [grams(t["question"])] + [grams(s) for s in sentences(t["question"])]) for t in tests]
    semantic = load_leak_decisions(decisions_path) if decisions_path else {}

    kept: list[dict[str, Any]] = []
    kept_grams: list[set[str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        q, a = (row.get("instruction") or "").strip(), (row.get("output") or "").strip()
        rid = row.get("id", "")
        reason = None
        if not q or not a:
            reason = "пустой вопрос или ответ"
        elif rid in seen_ids:
            reason = "повтор id"
        else:
            g = grams(q)
            leak = max(((tid, max(jaccard(g, x) for x in tgs)) for tid, tgs in test_grams),
                       key=lambda x: x[1], default=("", 0.0))
            dup = max((jaccard(g, kg) for kg in kept_grams), default=0.0)
            if rid in semantic:
                reason = f"тот же вопрос, что отложенный {semantic[rid]} (по смыслу)"
            elif leak[1] >= LEAK_THRESHOLD:
                reason = f"похож на отложенный {leak[0]} ({leak[1]:.2f})"
            elif dup >= DUP_THRESHOLD:
                reason = f"повтор вопроса ({dup:.2f})"
            elif lint(q, a):
                reason = f"стиль: {','.join(lint(q, a))}"
            else:
                kept_grams.append(g)
        if reason:
            report["dropped"].append({"id": rid, "reason": reason, "question": q})
            continue
        seen_ids.add(rid)
        kept.append({"id": rid, "category": row.get("category", ""), "instruction": q, "output": a})

    lengths = sorted(len(r["output"]) for r in kept)
    report.update({
        "kept": len(kept),
        "per_category": dict(Counter(r["category"] for r in kept)),
        "answer_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]} if kept else {},
        "test_questions": len(tests),
        "thresholds": {"duplicate": DUP_THRESHOLD, "leak_vs_test": LEAK_THRESHOLD},
        "semantic_leaks_dropped": sum(1 for d in report["dropped"] if "по смыслу" in d["reason"]),
    })
    return kept, report


def main() -> None:
    kept, report = build(ROOT / "data" / "raw", ROOT / "data" / "test_questions.json",
                         ROOT / "data" / "leak_decisions.json")
    (ROOT / "data" / "train.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "data" / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    print(f"партий: {report['raw']} пар, в датасет: {report['kept']}, отброшено: {len(report['dropped'])}")
    for d in report["dropped"]:
        print(f"  - {d['id']}: {d['reason']} | {d['question'][:80]}")
    print("по темам:", report["per_category"], "| длина ответа:", report["answer_chars"])


if __name__ == "__main__":
    main()
