import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fact_check import check  # noqa: E402


def test_numbers_from_facts_and_question_are_allowed():
    ans = "За октябрь помощник завёл 164 черновика, цену нашёл у 114, голосовое распознаёт за 1,5 секунды."
    assert check("Как дела у кейса?", ans)["numbers"] == []
    assert check("У нас 2000 объектов, потянете?", "Про 2000 объектов замеров нет.")["numbers"] == []


def test_invented_numbers_are_flagged():
    res = check("Как быстро появляется черновик?", "Черновиков было 386, и 379 из них прошли проверку за 2 минуты.")
    assert res["numbers"] == ["386", "379"]
    assert check("Погода?", "В январе 17-25 градусов.")["numbers"] == ["17", "25"]
    # 20 есть в фактах (часы ответа до 20:00), поэтому такое число проверка не ловит: она грубая
    assert check("Погода?", "В январе 18-20 градусов.")["numbers"] == ["18"]


def test_contacts():
    ok = "Пишите в Telegram @alexandero_optimum, t.me/alexandero_optimum, на alexandero.mobile@gmail.com или https://ai.optimum-ltd.com/."
    assert check("Как связаться?", ok)["contacts"] == []
    bad = "Почта alexandero.yandex.optimum@gmail.com, сайт https://ai.gpt.ru и @optimum_support."
    assert check("Как связаться?", bad)["contacts"] == ["alexandero.yandex.optimum@gmail.com", "https://ai.gpt.ru", "@optimum_support"]


def test_review_fixes_contacts():
    res = check("Как связаться?", "Пишите на alex@ai-optimum.ru или заходите на optimum-realty.ru, звоните +7 900 123-45-67.")
    assert res["contacts"] == ["+7 900 123-45-67", "alex@ai-optimum.ru", "optimum-realty.ru"]
    assert res["numbers"] == []
    assert check("Сайт?", "Сайт ai.optimum-ltd.com, а почта alexandero.mobile@gmail.com.")["contacts"] == []
