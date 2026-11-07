import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from style_lint import lint  # noqa: E402

GOOD = (
    "Может, если ей не мешать, поэтому цену проверяет код, а не нейросеть. В кейсе с переводом сайта на 13 языков "
    "каждое длинное число в переводе сверяется с оригиналом: если цена потерялась, перевод не сохраняется. "
    "А черновик объявления всё равно публикует сотрудник агентства. Хотите, Александр покажет это на бесплатном разборе?"
)


def test_example_from_style_guide_passes():
    assert lint("А нейросеть не будет выдумывать цены?", GOOD) == []


def test_markdown_and_lists():
    assert "markup" in lint("Что такое RAG?", "**RAG** это поиск.\n- первое\n- второе\n" + GOOD)
    assert "markup" in lint("Что такое RAG?", "## Ответ\n" + GOOD)


def test_length_depends_on_question():
    assert "length" in lint("Как работает подбор квартир?", "Через бота. Напишите?")
    assert lint("Спасибо", "Пожалуйста. Если появится задача, напишите Александру в Telegram @alexandero_optimum.") == []


def test_exclaim_emoji_opener_hype():
    assert "exclaim" in lint("Вопрос", GOOD.replace("код,", "код!"))
    assert "exclaim" in lint("Вопрос", GOOD + " 🙂")
    assert "opener" in lint("Вопрос", "Отличный вопрос. " + GOOD)
    assert "hype" in lint("Вопрос", GOOD.replace("код,", "уникальный код,"))
    assert "hype" in lint("Вопрос", GOOD.replace("код,", "код, это гарантирует точность,"))


def test_ty_and_capital_vy():
    assert "ty" in lint("Вопрос", GOOD.replace("Хотите,", "Хочешь, тебе"))
    assert "ty" in lint("Вопрос", GOOD.replace("Хотите,", "Если Вам удобно, хотите,"))
    assert "ty" not in lint("Вопрос", "Вы правы, это частая задача. " + GOOD)


def test_typography():
    assert "typo" in lint("Вопрос", GOOD.replace("Может,", "Может — "))
    assert "typo" in lint("Вопрос", GOOD.replace("цена потерялась", "«цена» потерялась"))


def test_price():
    assert "price" in lint("Сколько стоит?", GOOD.replace("Хотите,", "Обычно это 50 000 рублей. Хотите,"))
    assert "price" in lint("Сколько стоит?", GOOD.replace("Хотите,", "Около $500. Хотите,"))
    assert "price" not in lint("Вопрос", GOOD.replace("Хотите,", "Например, двушка до 60 тысяч долларов. Хотите,"))
    assert "price" not in lint("Бюджет 80 тысяч долларов, что найдёте?", GOOD.replace("Хотите,", "Бюджет 80 тысяч долларов бот учтёт. Хотите,"))


def test_next_step():
    assert "next_step" in lint("Вопрос", GOOD.rsplit(" Хотите", 1)[0])
    assert lint("Вопрос", GOOD.rsplit(" Хотите", 1)[0] + " Опишите вашу задачу в двух предложениях.") == []


def test_review_fixes():
    base = GOOD.rsplit(" Хотите", 1)[0]
    assert lint("Вопрос", base + " Скажите ему, что клиенты пишут ночью.") == []
    assert "ty" not in lint("Вопрос", base + ' Например, промпт "ты помощник агентства" тоже работает. Опишите задачу.')
    assert "ty" in lint("Вопрос", base + " Если тебе удобно, опишите задачу.")
    assert "price" in lint("Вопрос", GOOD.replace("Хотите,", "Квартира от 8 млн руб. Хотите,"))
    assert "price" in lint("Вопрос", GOOD.replace("Хотите,", "Бюджет 2 миллиона рублей. Хотите,"))
