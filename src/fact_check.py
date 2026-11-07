"""Грубая автоматическая проверка «не из справки» — сигнал для судей и человека, а не счёт выдумок.

- числа больше 3, которых нет ни в фактах (style/FACTS.md, style/facts_prompt.txt), ни в вопросе: ловит «386 черновиков»
  вместо 164, но и нумерацию списков, «ФЗ-152», «GPT-3.5», примеры вроде «квартира 50 м²»;
- контакты и ссылки, которых нет в фактах: почта, адреса сайтов (и без http), ники Telegram, телефоны —
  в том числе настоящие внешние сайты, которые выдумкой не являются.

Смысловые выдумки («передам вопрос Александру») так не поймать — их считают судьи (src/judge.py).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)[^\s,;()\"'«»<>\[\]]+|\bt\.me/\w+", re.IGNORECASE)
DOMAIN = re.compile(r"(?<![\w@./-])(?:[a-z0-9][a-z0-9-]*\.)+(?:ru|com|ai|io|net|org|me|рф)(?![\w-])", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")
HANDLE = re.compile(r"(?<![\w.])@[A-Za-z][\w_]{3,}")
ALLOWED_CONTACTS = {"alexandero.mobile@gmail.com", "@alexandero_optimum", "t.me/alexandero_optimum",
                    "ai.optimum-ltd.com"}


def norm_number(text: str) -> str:
    return text.replace(",", ".").lstrip("0") or "0"


@lru_cache(maxsize=1)
def fact_numbers() -> frozenset[str]:
    text = "".join((ROOT / "style" / name).read_text(encoding="utf-8") for name in ("FACTS.md", "facts_prompt.txt"))
    return frozenset(norm_number(n) for n in NUMBER.findall(text))


def unsupported_numbers(question: str, answer: str) -> list[str]:
    allowed = fact_numbers() | {norm_number(n) for n in NUMBER.findall(question)}
    clean = PHONE.sub(" ", EMAIL.sub(" ", URL.sub(" ", answer)))
    out = []
    for n in NUMBER.findall(clean):
        value = norm_number(n)
        if value not in allowed and float(value) > 3:
            out.append(n)
    return out


def unknown_contacts(answer: str) -> list[str]:
    """Почта, ссылки, голые домены, ники Telegram и телефоны, которых нет в фактах."""
    rest = EMAIL.sub(" ", URL.sub(" ", answer))  # голые домены ищутся после почты и ссылок, без повторов
    found = EMAIL.findall(answer) + URL.findall(answer) + DOMAIN.findall(rest) + HANDLE.findall(answer)
    out = [p.strip() for p in PHONE.findall(answer)]  # телефонов в фактах нет: любой — чужой или выдуманный
    for item in found:
        key = item.lower().rstrip("/.?!:;»)")
        key = re.sub(r"^(https?://)?(www\.)?", "", key)
        if not any(key == a or key.startswith(a + "/") for a in ALLOWED_CONTACTS):
            out.append(item)
    return out


def check(question: str, answer: str) -> dict[str, list[str]]:
    return {"numbers": unsupported_numbers(question, answer), "contacts": unknown_contacts(answer)}
