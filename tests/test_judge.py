import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import judge  # noqa: E402


def test_packets_per_judge_and_unblinding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge, "JUDGE_DIR", tmp_path)
    rows = []
    for i in range(1, 9):
        for mode in ("lora", "base"):
            rows.append({"id": f"t-{i}", "mode": mode, "question": f"q{i}", "answer": f"{mode}-{i}"})
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"meta": {}, "answers": rows}), encoding="utf-8")
    judge.make_packets(answers, [("lora", "base")], judges=3)

    keys = [json.loads((tmp_path / f"lora_vs_base.p{k}_key.json").read_text(encoding="utf-8")) for k in (1, 2, 3)]
    packets = [json.loads((tmp_path / f"lora_vs_base.p{k}.json").read_text(encoding="utf-8")) for k in (1, 2, 3)]
    assert all(set(item) == {"id", "question", "A", "B"} for p in packets for item in p)  # режимы не названы
    assert keys[0] != keys[1] or keys[1] != keys[2]  # у судей разный порядок букв

    # Все судьи выбирают ответ lora; двое из трёх видят выдумку в ответе base.
    for k, (packet, flag) in enumerate(zip(packets, [True, True, False]), 1):
        verdicts = []
        for item in packet:
            lora_letter = "A" if item["A"].startswith("lora") else "B"
            base_letter = "B" if lora_letter == "A" else "A"
            verdicts.append({"id": item["id"], "style": lora_letter, "helpful": "tie", "overall": lora_letter,
                             f"{base_letter}_invented": flag, f"{lora_letter}_invented": False})
        (tmp_path / f"lora_vs_base.judge{k}.json").write_text(json.dumps(verdicts), encoding="utf-8")
    res = judge.report([("lora", "base")])["lora_vs_base"]
    assert res["style"] == {"lora": 8, "base": 0, "tie": 0}
    assert res["helpful"]["tie"] == 8
    assert res["invented_facts"] == {"lora": 0, "base": 8}
