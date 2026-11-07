#!/usr/bin/env bash
# Проверка без GPU: типы (basedpyright, только ошибки) и тесты линтера, проверки выдумок, сборки датасета и судей.
# Сеть нужна только при первом запуске: uv скачает basedpyright и pytest, дальше — из кэша.
set -euo pipefail
cd "$(dirname "$0")"
uvx basedpyright --level error src/style_lint.py src/prompting.py src/evaluate.py src/build_dataset.py \
    src/fact_check.py src/judge.py src/semantic_leak.py src/ollama_answers.py src/make_chart.py
uv run --with pytest python -m pytest -q tests
python src/style_lint.py data/train.json
