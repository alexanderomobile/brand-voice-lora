"""Дообучение LoRA-адаптера на парах «вопрос–ответ» (по мотивам fine_tuning/train.py из урока).

    python src/train_lora.py --model_name models/Qwen_Qwen3-4B-Instruct-2507 \
        --dataset_path data/train.json --output_dir runs/qwen3-4b-brand

Что изменено относительно кода урока и почему:
- шаблон диалога один для обучения и чата (src/prompting.py): в уроке обучение и чат подавали модели разный текст;
- loss только по ответу: вопрос и паддинг в labels = -100, в конце ответа стоит EOS. В уроке pad = eos,
  коллатор маскировал все EOS, и модель не училась останавливаться;
- динамический паддинг вместо padding="max_length" (меньше пустых вычислений);
- прогрев — доля от числа шагов: 100 шагов прогрева урока длиннее всего обучения на ~500 примерах с батчем 16;
- отложенная выборка, eval loss после каждой эпохи, журнал train.log и metrics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompting import has_chat_template, training_texts  # noqa: E402

log = logging.getLogger("train")

TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = [row for row in data if row.get("instruction") and row.get("output")]
    log.info("датасет %s: %d пар (пропущено пустых: %d)", path.name, len(pairs), len(data) - len(pairs))
    return pairs


def split_pairs(pairs: list[dict[str, Any]], val_ratio: float,
                seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Отложенная выборка с долей каждой темы (поле category, если есть)."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_cat.setdefault(row.get("category", "all"), []).append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for rows in by_cat.values():
        rows = rows[:]
        rng.shuffle(rows)
        k = round(len(rows) * val_ratio)
        val += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    return train, val


def tokenize_pairs(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> tuple[Dataset, int]:
    items, truncated = [], 0
    for row in rows:
        prompt, full = training_texts(tokenizer, row["instruction"], row["output"])
        # Промпт и ответ токенизируются отдельно: при генерации модель тоже получает промпт отдельно
        # и продолжает его, так что граница токенов в обучении и в работе одна и та же.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_length:
            truncated += 1
            full_ids = full_ids[:max_length]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        items.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels[: len(full_ids)]})
    return Dataset.from_list(items), truncated


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


class MetricsLog(TrainerCallback):
    """Пишет историю loss в metrics.json и короткие строки в журнал, как DetailedLoggingCallback урока."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.started = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        parts = [f"шаг {state.global_step}/{state.max_steps}", f"эпоха {state.epoch or 0:.2f}"]
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
            if key in logs:
                parts.append(f"{key}={logs[key]:.4g}")
        log.info(" | ".join(parts))
        self.path.write_text(json.dumps(state.log_history, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True, help="имя модели на Hugging Face или локальная папка")
    ap.add_argument("--dataset_path", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--num_train_epochs", type=float, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    setup_logging(args.output_dir)
    set_seed(args.seed)
    use_gpu = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_gpu:
        raise SystemExit("запрошен --device cuda, но CUDA недоступна")
    bf16 = use_gpu and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_gpu else torch.float32)
    gpu = torch.cuda.get_device_name(0) if use_gpu else "нет"
    log.info("torch %s, CUDA %s, GPU: %s, dtype %s", torch.__version__, torch.version.cuda, gpu, dtype)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    model.to("cuda" if use_gpu else "cpu")
    model_type = model.config.model_type
    log.info("модель %s (%s): %.0f млн параметров, шаблон: %s, загрузка %.1f с", args.model_name, model_type,
             sum(p.numel() for p in model.parameters()) / 1e6,
             "chat template модели" if has_chat_template(tokenizer) else "шаблон урока", time.time() - t0)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type=TaskType.CAUSAL_LM, target_modules=TARGET_MODULES.get(model_type, "all-linear"))
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("LoRA r=%d alpha=%d dropout=%.2f: обучается %.1f млн параметров (%.2f%%)", args.lora_r,
             args.lora_alpha, args.lora_dropout, trainable / 1e6, 100 * trainable / total)

    pairs = load_pairs(args.dataset_path)
    train_rows, val_rows = split_pairs(pairs, args.val_ratio, args.seed)
    train_ds, cut_train = tokenize_pairs(train_rows, tokenizer, args.max_length)
    val_ds, cut_val = tokenize_pairs(val_rows, tokenizer, args.max_length)
    lengths = [len(x) for x in train_ds["input_ids"]]
    log.info("обучение %d, проверка %d пар; длина в токенах: средняя %.0f, макс %d; обрезано %d",
             len(train_ds), len(val_ds), sum(lengths) / len(lengths), max(lengths), cut_train + cut_val)

    targs = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        logging_first_step=True,
        eval_strategy="epoch" if len(val_ds) else "no",
        save_strategy="no",
        bf16=bf16,
        fp16=use_gpu and not bf16,
        use_cpu=not use_gpu,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds if len(val_ds) else None,
                      data_collator=Collator(tokenizer.pad_token_id),
                      callbacks=[MetricsLog(args.output_dir / "metrics.json")])
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    result = trainer.train()
    train_s = time.time() - t1
    final_eval = trainer.evaluate() if len(val_ds) else {}
    peak = torch.cuda.max_memory_allocated() / 1024**3 if use_gpu else 0.0
    log.info("обучение заняло %.1f мин, train loss %.4f, eval loss %s, пик памяти GPU %.1f ГБ", train_s / 60,
             result.training_loss, final_eval.get("eval_loss"), peak)

    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    import peft
    import transformers
    summary = {
        "model_name": args.model_name, "model_type": model_type, "dataset": str(args.dataset_path),
        "dataset_sha256": hashlib.sha256(args.dataset_path.read_bytes()).hexdigest(),
        "pairs": len(pairs), "train_pairs": len(train_ds), "val_pairs": len(val_ds),
        "val_ids": [r.get("id") for r in val_rows], "args": {k: str(v) for k, v in vars(args).items()},
        "trainable_params": trainable, "total_params": total, "train_minutes": round(train_s / 60, 2),
        "train_loss": result.training_loss, "eval_loss": final_eval.get("eval_loss"), "steps": result.global_step,
        "gpu": gpu, "dtype": str(dtype), "peak_gpu_gb": round(peak, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
    log.info("адаптер сохранён: %s", args.output_dir / "adapter")


if __name__ == "__main__":
    main()
