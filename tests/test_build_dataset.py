import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_dataset import build  # noqa: E402
from test_style_lint import GOOD  # noqa: E402


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_dedup_leak_and_style(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "a.json", [
        {"id": "a-1", "instruction": "Можно ли автоматизировать перенос объявлений из чатов на сайт?", "output": GOOD, "category": "a"},
        {"id": "a-2", "instruction": "Можно ли автоматизировать перенос объявлений из чатов на сайт", "output": GOOD, "category": "a"},
        {"id": "a-3", "instruction": "Какая погода в Хургаде в январе?", "output": GOOD, "category": "a"},
        {"id": "a-4", "instruction": "Что такое токены?", "output": "**Токены** это кусочки текста!", "category": "a"},
        {"id": "a-5", "instruction": "", "output": GOOD, "category": "a"},
    ])
    test = tmp_path / "test.json"
    write(test, [{"id": "test-01", "question": "Какая погода в Хургаде в январе? Хочу прилететь", "category": "off"}])
    kept, report = build(raw, test)
    assert [r["id"] for r in kept] == ["a-1"]
    reasons = {d["id"]: d["reason"] for d in report["dropped"]}
    assert reasons["a-2"].startswith("повтор вопроса")
    assert reasons["a-3"].startswith("похож на отложенный test-01")
    assert reasons["a-4"].startswith("стиль")
    assert reasons["a-5"] == "пустой вопрос или ответ"


def test_sentence_level_and_semantic_leaks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "a.json", [
        {"id": "a-1", "instruction": "Наши клиенты терпеть не могут ботов и сразу просят живого менеджера", "output": GOOD, "category": "a"},
        {"id": "a-2", "instruction": "Сколько часов в неделю экономит помощник?", "output": GOOD, "category": "a"},
        {"id": "a-3", "instruction": "Какие языки знает бот?", "output": GOOD, "category": "a"},
    ])
    test = tmp_path / "test.json"
    write(test, [{"id": "test-01", "question": "Наши клиенты терпеть не могут ботов, сразу просят живого менеджера. "
                  "Не отпугнём ли мы их таким помощником? И сколько это стоит в месяц?", "category": "o"},
                 {"id": "test-02", "question": "Помощник реально экономит время? На сколько?", "category": "o"}])
    decisions = tmp_path / "decisions.json"
    write(decisions, [{"test_id": "test-02", "train_id": "a-2", "same": True},
                      {"test_id": "test-02", "train_id": "a-3", "same": False}])
    kept, report = build(raw, test, decisions)
    assert [r["id"] for r in kept] == ["a-3"]
    reasons = {d["id"]: d["reason"] for d in report["dropped"]}
    assert reasons["a-1"].startswith("похож на отложенный test-01")
    assert "по смыслу" in reasons["a-2"]
