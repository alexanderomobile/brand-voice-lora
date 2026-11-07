"""Слепое попарное сравнение ответов LLM-судьями.

    python src/judge.py packets eval/answers_qwen3-4b.json --judges 3
    python src/judge.py report

packets — для каждой пары режимов и каждого судьи k пишет eval/judge/<a>_vs_<b>.p<k>.json: вопрос и два ответа
под буквами A и B. Какой режим под какой буквой, решает хеш (пара, k, id вопроса), так что у разных судей порядок
разный и сдвиг к первой позиции не накапливается. Ключ — отдельно в *.p<k>_key.json, судьям не показывается.
Инструкция судьям — eval/judge/prompt.md.
report — собирает вердикты eval/judge/<a>_vs_<b>.judge<k>.json (судья k оценивал пакет p<k>): по каждому
критерию большинство голосов, итог «победы / ничьи / поражения» режима a и число ответов с выдумкой у каждого
режима (выдумка засчитывается, если её отметило большинство судей). Итог дописывается в eval/judge/report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "eval" / "judge"
CRITERIA = ("style", "helpful", "overall")
DEFAULT_PAIRS = "lora_facts:base_prompt,lora_facts:lora,lora:base"


def a_first(qid: str, pair: str, judge: int) -> bool:
    return hashlib.sha256(f"{pair}:{judge}:{qid}".encode()).digest()[0] % 2 == 0


def make_packets(answers_path: Path, pairs: list[tuple[str, str]], judges: int = 3) -> None:
    rows = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
    by = {(r["mode"], r["id"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        name = f"{a}_vs_{b}"
        for k in range(1, judges + 1):
            items, key = [], {}
            for qid in ids:
                ra, rb = by[(a, qid)], by[(b, qid)]
                first = a_first(qid, name, k)
                items.append({"id": qid, "question": ra["question"],
                              "A": (ra if first else rb)["answer"], "B": (rb if first else ra)["answer"]})
                key[qid] = {"A": a if first else b, "B": b if first else a}
            (JUDGE_DIR / f"{name}.p{k}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            (JUDGE_DIR / f"{name}.p{k}_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
            print(f"{name} p{k}: {len(items)} вопросов, {sum(v['A'] == a for v in key.values())} раз {a} под буквой A")


def report(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    path = JUDGE_DIR / "report.json"
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for a, b in pairs:
        name = f"{a}_vs_{b}"
        verdicts: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = {}
        judges = sorted(JUDGE_DIR.glob(f"{name}.judge*.json"))
        for vpath in judges:
            k = vpath.name.split(".judge")[1].split(".")[0]
            key = json.loads((JUDGE_DIR / f"{name}.p{k}_key.json").read_text(encoding="utf-8"))
            for v in json.loads(vpath.read_text(encoding="utf-8")):
                verdicts.setdefault(v["id"], []).append((v, key[v["id"]]))
        tally: dict[str, Counter[str]] = {c: Counter() for c in CRITERIA}
        invented: Counter[str] = Counter()
        for vs in verdicts.values():
            for c in CRITERIA:
                votes = Counter(k[v[c]] if v[c] in ("A", "B") else "tie" for v, k in vs)
                top, n = votes.most_common(1)[0]
                tally[c]["tie" if n * 2 <= len(vs) else top] += 1
            for mode in (a, b):
                flagged = sum(bool(v.get(f"{letter}_invented")) for v, k in vs for letter in ("A", "B")
                              if k[letter] == mode)
                if flagged * 2 > len(vs):
                    invented[mode] += 1
        out[name] = {"judges": len(judges), "questions": len(verdicts),
                     **{c: {a: tally[c][a], b: tally[c][b], "tie": tally[c]["tie"]} for c in CRITERIA},
                     "invented_facts": {a: invented[a], b: invented[b]}}
        print(f"{name}: судей {len(judges)}, вопросов {len(verdicts)}")
        for c in CRITERIA:
            print(f"  {c:8s} {a} {tally[c][a]:2d} | ничья {tally[c]['tie']:2d} | {b} {tally[c][b]:2d}")
        print(f"  выдумки (большинство судей): {a} {invented[a]}, {b} {invented[b]}")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["packets", "report"])
    ap.add_argument("answers", type=Path, nargs="?")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--judges", type=int, default=3)
    args = ap.parse_args()
    pairs = [(p.split(":")[0], p.split(":")[1]) for p in args.pairs.split(",")]
    if args.command == "packets":
        if not args.answers:
            raise SystemExit("нужен файл ответов")
        make_packets(args.answers, pairs, args.judges)
    else:
        report(pairs)


if __name__ == "__main__":
    main()
