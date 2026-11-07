"""Автоматическая проверка ответа на правила стиля из style/STYLE.md (пометка [авто]).

    python src/style_lint.py data/train.json            # сводка по датасету
    python src/style_lint.py eval/answers.json --field answer --show 5

Файл — JSON-список объектов с вопросом (instruction/question) и ответом (output/answer).
Каждое правило возвращает код; ответ проходит стиль, если кодов нет.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

RULES = {
    "markup": "1 разметка: **, #, `, таблица, список или больше трёх абзацев",
    "length": "2 длина вне диапазона",
    "exclaim": "3 восклицательный знак или эмодзи",
    "opener": "4 пустое вступление",
    "hype": "5 рекламный штамп или гарантия",
    "ty": "6 обращение на «ты» или «Вы» с большой буквы",
    "typo": "7 «ёлочки», длинное или среднее тире, многоточие одним символом",
    "price": "8 цена или сумма денег",
    "next_step": "9 в конце нет вопроса или следующего шага",
}

SHORT_Q = re.compile(
    r"^\W*(привет\w*|здравствуй\w*|добр\w+ (день|утро|вечер)|доброго времени суток|hi|hello|спасибо\w*|"
    r"благодар\w*|ок\w*|понятно|ясно|хорошо|пока|до свидания|всего доброго|ага|угу|супер|класс)\b",
    re.IGNORECASE,
)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️⭐✅]")
OPENERS = re.compile(
    r"^\W*(отличный вопрос|хороший вопрос|прекрасный вопрос|интересный вопрос|конечно|безусловно|разумеется|"
    r"да,? конечно|как ии|как искусственный интеллект|я рад|рад помочь|спасибо за (ваш )?вопрос)",
    re.IGNORECASE,
)
HYPE = re.compile(
    r"уникальн|революцион|инновацион|под ключ|в кратчайшие сроки|индивидуальн\w* подход|эффективн\w* решени|"
    r"широк\w* спектр|лучш\w* на рынке|высококвалифицирован|гарантир|гаранти[яи]|100\s?%",
    re.IGNORECASE,
)
TY = re.compile(r"\b(ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твоих|твоим)\b", re.IGNORECASE)
CAP_VY = re.compile(r"\b(Вы|Вас|Вам|Вами|Ваш\w*)\b")
TYPO = re.compile("[«»—–…]")
MONEY = re.compile(
    r"\d[\d\s.,]*\s*((тыс\w*|млн|млрд|миллион\w*|миллиард\w*)\.?\s*)?(₽|руб\w*|р\.|\$|долл\w*|usd|евро|€|eur)"
    r"|[₽$€]\s*\d",
    re.IGNORECASE,
)
MONEY_OK = re.compile(r"до 60 тысяч долларов", re.IGNORECASE)
NEXT_STEP = re.compile(
    r"напишите|опишите|расскажите|пришлите|оставьте|запишитесь|задайте|спросите|обращайтесь|приходите|"
    r"посмотрите|загляните|начните|скажите|назовите|перепишите|сообщите|уточните|покажите|"
    r"@alexandero_optimum|разбор|созвон|форм\w* на сайте",
    re.IGNORECASE,
)
QUOTED = re.compile(r'"[^"\n]*"|«[^»\n]*»')  # «ты» в цитате (пример промпта) — не обращение к собеседнику
LIST_LINE = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.MULTILINE)


def is_short_question(question: str) -> bool:
    q = question.strip()
    return len(q) <= 40 and bool(SHORT_Q.match(q))


def capital_vy_mid_sentence(text: str) -> bool:
    """«Вы» с большой буквы не в начале предложения — вежливая форма письма, а не мессенджера."""
    for m in CAP_VY.finditer(text):
        before = text[: m.start()].rstrip(" \"'(")
        if before and before[-1] not in ".!?:\n":
            return True
    return False


def last_sentence(text: str) -> str:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts[-1] if parts else ""


def lint(question: str, answer: str) -> list[str]:
    """Коды нарушенных правил (пустой список — ответ в стиле)."""
    a = answer.strip()
    codes: list[str] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", a) if p.strip()]
    if "**" in a or "`" in a or re.search(r"^#{1,6}\s", a, re.M) or re.search(r"^\|.*\|\s*$", a, re.M) \
            or LIST_LINE.search(a) or len(paragraphs) > 3:
        codes.append("markup")
    lo, hi = (20, 300) if is_short_question(question) else (150, 700)
    if not lo <= len(a) <= hi:
        codes.append("length")
    if "!" in a or EMOJI.search(a):
        codes.append("exclaim")
    if OPENERS.match(a):
        codes.append("opener")
    if HYPE.search(a):
        codes.append("hype")
    if TY.search(QUOTED.sub("", a)) or capital_vy_mid_sentence(a):
        codes.append("ty")
    if TYPO.search(a):
        codes.append("typo")
    money = [m.group(0) for m in MONEY.finditer(MONEY_OK.sub("", a))]
    if money and not any(m.strip() in question for m in money):
        codes.append("price")
    tail = last_sentence(a)
    if not (tail.endswith("?") or NEXT_STEP.search(tail)):
        codes.append("next_step")
    return codes


def load_pairs(path: Path, field: str | None) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for row in data:
        q = row.get("instruction") or row.get("question") or ""
        a = row.get(field) if field else (row.get("output") or row.get("answer"))
        pairs.append((q, a or ""))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--field", help="поле с ответом (по умолчанию output или answer)")
    ap.add_argument("--show", type=int, default=0, help="показать N примеров с нарушениями")
    args = ap.parse_args()
    pairs = load_pairs(args.path, args.field)
    per_rule: Counter[str] = Counter()
    bad = []
    for i, (q, a) in enumerate(pairs):
        codes = lint(q, a)
        per_rule.update(codes)
        if codes:
            bad.append((i, codes, q, a))
    n = len(pairs)
    ok = n - len(bad)
    print(f"{args.path.name}: {ok} из {n} в стиле ({ok / n:.0%})" if n else "пусто")
    for code, text in RULES.items():
        if per_rule[code]:
            print(f"  {per_rule[code]:4d}  {text}")
    for i, codes, q, a in bad[: args.show]:
        print(f"\n#{i} {codes}\nQ: {q}\nA: {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
