"""Сводка по ответам на отложенные вопросы: стиль по линтеру, длина, промпт, скорость.

    python src/evaluate.py eval/answers_qwen3-4b.json eval/answers_rugpt3small.json --out eval/style_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_check import check  # noqa: E402
from style_lint import RULES, lint  # noqa: E402


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_rule: Counter[str] = Counter()
    passed = passed_no_typo = invented_numbers = unknown_contacts = 0
    for r in rows:
        codes = lint(r["question"], r["answer"])
        r["lint"] = codes
        r["fact_check"] = check(r["question"], r["answer"])
        per_rule.update(codes)
        passed += not codes
        passed_no_typo += not [c for c in codes if c != "typo"]
        invented_numbers += bool(r["fact_check"]["numbers"])
        unknown_contacts += bool(r["fact_check"]["contacts"])
    n = len(rows)
    lengths = [len(r["answer"]) for r in rows]
    tok_s = [r["new_tokens"] / r["seconds"] for r in rows if r["seconds"] > 0]
    return {
        "answers": n,
        "style_pass": passed,
        "style_pass_pct": round(100 * passed / n, 1) if n else 0.0,
        "style_pass_without_typography": passed_no_typo,
        "violations": {code: per_rule[code] for code in RULES if per_rule[code]},
        "answers_with_unsupported_numbers": invented_numbers,
        "answers_with_unknown_contacts": unknown_contacts,
        # обрезка известна только для ответов из generate_answers.py (поле truncated); у chat.py урока её нет
        "truncated": sum(1 for r in rows if r.get("truncated")) if all("truncated" in r for r in rows) else None,
        "answer_chars_median": statistics.median(lengths) if lengths else 0,
        "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] for r in rows), 1) if rows else 0,
        "new_tokens_mean": round(statistics.mean(r["new_tokens"] for r in rows), 1) if rows else 0,
        "seconds_median": round(statistics.median(r["seconds"] for r in rows), 2) if rows else 0,
        "tokens_per_second_median": round(statistics.median(tok_s), 1) if tok_s else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report: dict[str, Any] = {}
    for path in args.files:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = path.stem.removeprefix("answers_")
        modes: dict[str, list[dict[str, Any]]] = {}
        for row in data["answers"]:
            modes.setdefault(row["mode"], []).append(row)
        report[model] = {mode: summarize(rows) for mode, rows in modes.items()}
        for mode, s in report[model].items():
            print(f"{model:40s} {mode:12s} стиль {s['style_pass']:3d}/{s['answers']} ({s['style_pass_pct']:5.1f}%)"
                  f"  медиана {s['answer_chars_median']:5.0f} зн.  промпт {s['prompt_tokens_mean']:6.0f} ток."
                  f"  {s['tokens_per_second_median']:5.1f} ток/с  без типографики {s['style_pass_without_typography']}"
                  f"  выдуманные числа {s['answers_with_unsupported_numbers']}, контакты {s['answers_with_unknown_contacts']}"
                  f"  нарушения {s['violations']}")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")  # с кодами lint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
