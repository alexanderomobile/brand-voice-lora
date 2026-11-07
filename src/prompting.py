"""Один шаблон диалога для обучения, чата и оценки.

В коде урока обучение шло на тексте "### Instruction: ... ### Response: ...", а чат подавал модели
"User: ... Assistant:" — модель видела в работе не тот формат, которому училась. Здесь оба места берут
текст отсюда: у instruct-моделей это их собственный chat template, у моделей без него (GPT-2, ruGPT-3) —
шаблон урока.
"""
from __future__ import annotations

from typing import Any

LESSON_TEMPLATE = "### Instruction:\n{question}\n\n### Response:\n"


def has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_messages(question: str, system: str | None = None,
                   history: list[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for q, a in history or []:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


def prompt_text(tokenizer: Any, question: str, system: str | None = None,
                history: list[tuple[str, str]] | None = None) -> str:
    """Текст, после которого модель должна начать ответ."""
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(build_messages(question, system, history),
                                             tokenize=False, add_generation_prompt=True)
    parts = [f"{system}\n\n"] if system else []
    for q, a in history or []:
        parts.append(LESSON_TEMPLATE.format(question=q) + a + "\n\n")
    parts.append(LESSON_TEMPLATE.format(question=question))
    return "".join(parts)


def training_texts(tokenizer: Any, question: str, answer: str) -> tuple[str, str]:
    """(промпт, полный текст с ответом и концом реплики). Loss считается только по хвосту после промпта."""
    prompt = prompt_text(tokenizer, question)
    if has_chat_template(tokenizer):
        messages = build_messages(question) + [{"role": "assistant", "content": answer}]
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        if full.startswith(prompt):
            return prompt, full.rstrip("\n")
    return prompt, prompt + answer + tokenizer.eos_token
