#!/usr/bin/env python3
"""QLoRA fine-tuning for *text* drawing-copilot behavior.

This trains a Transformers-format causal language model on curated JSONL
``messages`` examples. It does not accept Ollama GGUF files and does not train
the primitive detector or vision encoder. Use a smaller text base model for an
8 GB GPU, then keep qwen3.5:9b for vision/tool-grounded inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from cli_progress import RichProgress, add_progress_argument


REQUIRED = ("accelerate", "bitsandbytes", "datasets", "peft", "transformers")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a text copilot adapter with 4-bit QLoRA.")
    parser.add_argument("--base-model", required=True, help="Local path or Hugging Face Transformers-format text causal-LM checkpoint; not an Ollama GGUF tag.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL messages file made by build_copilot_training_data.py.")
    parser.add_argument("--output", type=Path, default=Path("adapters/drawing-copilot-lora"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048, help="Use 1024-2048 on an 8 GB GPU.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    add_progress_argument(parser)
    args = parser.parse_args()
    if not args.dataset.is_file():
        print(f"[ERROR] Dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print("[ERROR] Missing fine-tuning dependencies: " + ", ".join(missing), file=sys.stderr)
        print("Install with: python -m pip install -r requirements-copilot-training.txt", file=sys.stderr)
        return 2
    try:
        import torch
        from datasets import disable_progress_bar, load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments
        from transformers.trainer_callback import PrinterCallback
    except ImportError as exc:  # pragma: no cover - guarded above
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[ERROR] QLoRA training requires a CUDA GPU.", file=sys.stderr)
        return 2
    progress = RichProgress(enabled=not args.no_progress).start()
    setup_task = progress.add_task("Preparing QLoRA training", total=4)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    progress.advance(setup_task)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quantization, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"),
    )
    progress.advance(setup_task)
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    progress.advance(setup_task)

    def tokenize(row: dict[str, object]) -> dict[str, object]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each training row must contain a messages array.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=args.max_length)

    disable_progress_bar()
    tokenization_task = progress.add_task("Tokenizing copilot training data", total=1)
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    progress.complete(tokenization_task, f"Tokenized {len(tokenized):,} examples")
    progress.complete(setup_task, "QLoRA training prepared")
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to=[],
        optim="paged_adamw_8bit",
        disable_tqdm=True,
    )

    class RichTrainingCallback(TrainerCallback):
        def __init__(self, reporter: RichProgress) -> None:
            self.reporter = reporter
            self.task_id: int | None = None
            self.last_loss: float | None = None

        def on_train_begin(self, _args: object, state: object, control: object, **_kwargs: object) -> object:
            total_steps = max(1, int(getattr(state, "max_steps", 1) or 1))
            self.task_id = self.reporter.add_task("QLoRA fine-tuning", total=total_steps)
            return control

        def on_log(self, _args: object, state: object, control: object, logs: dict[str, float] | None = None, **_kwargs: object) -> object:
            if logs and "loss" in logs:
                self.last_loss = float(logs["loss"])
                self.reporter.update(self.task_id, description=f"QLoRA fine-tuning: loss {self.last_loss:.4f}")
            return control

        def on_step_end(self, _args: object, state: object, control: object, **_kwargs: object) -> object:
            self.reporter.update(self.task_id, completed=int(getattr(state, "global_step", 0)))
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.remove_callback(PrinterCallback)
    callback = RichTrainingCallback(progress)
    trainer.add_callback(callback)
    trainer.train()
    progress.complete(callback.task_id, "QLoRA fine-tuning complete")
    save_task = progress.add_task("Saving LoRA adapter", total=1)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    progress.complete(save_task, "LoRA adapter saved")
    progress.stop()
    print(f"[DONE] Saved LoRA adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
