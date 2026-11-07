"""График итогов для README: стиль и выдумки по режимам на 60 отложенных вопросах.

    python src/make_chart.py            # пишет assets/results.svg

Данные берутся из eval/answers_*.json после src/evaluate.py (поля lint и fact_check).
Форма — выделение: рекомендованная схема синим, остальное серым; две панели по одним строкам
вместо второй оси. Светлая и тёмная тема — через prefers-color-scheme внутри SVG.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ROWS = [  # (подпись, файл, режим, выделить)
    ("ruGPT-3 small, код урока как есть", "answers_lesson_rugpt3small.json", "lesson", False),
    ("Qwen3-4B без дообучения", "answers_qwen3-4b.json", "base", False),
    ("Qwen3-4B, длинный промпт: стиль и факты", "answers_qwen3-4b.json", "base_prompt", False),
    ("Qwen3-4B, дообучение без фактов", "answers_qwen3-4b.json", "lora", False),
    ("Qwen3-4B, дообучение и справка с фактами", "answers_qwen3-4b.json", "lora_facts", True),
    ("то же в Ollama, GGUF Q4_K_M", "answers_ollama-q4.json", "ollama_facts", False),
]
W, LABEL_W, PANEL_W, GAP = 940, 330, 265, 40
ROW_H, BAR_H, TOP, MAX = 34, 16, 118, 60


def counts() -> list[tuple[str, int, int, bool]]:
    out = []
    for label, name, mode, accent in ROWS:
        rows = [r for r in json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))["answers"]
                if r["mode"] == mode]
        style = sum(1 for r in rows if not r["lint"])
        invented = sum(1 for r in rows if r["fact_check"]["numbers"] or r["fact_check"]["contacts"])
        out.append((label, style, invented, accent))
    return out


def bar(x0: float, y: float, value: int, accent: bool) -> str:
    """Горизонтальный столбик: квадратный у оси, скруглённый на конце (4px)."""
    w = PANEL_W * value / MAX
    cls = "accent" if accent else "muted"
    if w <= 0:
        return f'<rect class="base" x="{x0:.1f}" y="{y:.1f}" width="2" height="{BAR_H}"/>'
    r = min(4.0, w)
    return (f'<path class="{cls}" d="M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{BAR_H - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def panel(x0: float, title: str, values: list[tuple[int, bool]]) -> list[str]:
    parts = [f'<text class="h" x="{x0}" y="{TOP - 34}">{escape(title)}</text>']
    bottom = TOP + ROW_H * len(values)
    for tick in (0, 20, 40, 60):
        x = x0 + PANEL_W * tick / MAX
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP - 14}" x2="{x:.1f}" y2="{bottom - 8}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 8}" text-anchor="middle">{tick}</text>')
    for i, (value, accent) in enumerate(values):
        y = TOP + ROW_H * i - BAR_H / 2 - 4
        parts.append(bar(x0, y, value, accent))
        tx = x0 + PANEL_W * value / MAX + 6
        parts.append(f'<text class="{"val strong" if accent else "val"}" x="{tx:.1f}" y="{y + BAR_H - 3:.1f}">{value}</text>')
    return parts


def main() -> None:
    data = counts()
    h = TOP + ROW_H * len(data) + 64
    p1 = LABEL_W
    p2 = LABEL_W + PANEL_W + GAP
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        'role="img" aria-labelledby="t d">',
        '<title id="t">Стиль бренда и числа не из справки: 60 отложенных вопросов</title>',
        '<desc id="d">' + escape("; ".join(f"{l}: в стиле {s} из 60, с числами или контактами не из справки {v}"
                                           for l, s, v, _ in data)) + "</desc>",
        """<style>
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;--accent:#2a78d6}
@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--base:#383835;--accent:#3987e5}}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink2);font-size:13px}
.title{fill:var(--ink);font-size:17px;font-weight:600}.sub{font-size:12.5px}.h{fill:var(--ink);font-size:13px;font-weight:600}
.lab{fill:var(--ink);font-size:13px}.lab.strong,.val.strong{font-weight:600;fill:var(--ink)}.val{font-variant-numeric:tabular-nums}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.note{fill:var(--muted);font-size:11.5px}
.grid{stroke:var(--grid);stroke-width:1}.accent{fill:var(--accent)}.muted{fill:var(--muted)}.base{fill:var(--base)}
.bg{fill:var(--surface)}
</style>""",
        f'<rect class="bg" width="{W}" height="{h}" rx="10"/>',
        '<text class="title" x="24" y="34">Стиль держит дообучение, цифры и контакты держит справка</text>',
        '<text class="sub" x="24" y="55">60 отложенных вопросов, найденные пересказы убраны из обучения; Qwen3-4B-Instruct-2507, LoRA на 385 парах</text>',
    ]
    for i, (label, _, _, accent) in enumerate(data):
        y = TOP + ROW_H * i
        parts.append(f'<text class="{"lab strong" if accent else "lab"}" x="24" y="{y:.1f}">{escape(label)}</text>')
    parts += panel(p1, "Ответов в стиле бренда (из 60)", [(s, a) for _, s, _, a in data])
    parts += panel(p2, "Числа, контакты не из справки", [(v, a) for _, _, v, a in data])
    note_y = TOP + ROW_H * len(data) + 32
    parts.append(f'<text class="note" x="24" y="{note_y}">Стиль: линтер style/STYLE.md, все 9 правил. Длинный промпт без учёта типографики проходит 35 из 60: длинное тире</text>')
    parts.append(f'<text class="note" x="24" y="{note_y + 16}">осталось во всех 60 ответах. Числа и контакты — грубая автопроверка с ложными срабатываниями (src/fact_check.py); выдумки по смыслу считали судьи.</text>')
    parts.append("</svg>")
    out = ROOT / "assets" / "results.svg"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
