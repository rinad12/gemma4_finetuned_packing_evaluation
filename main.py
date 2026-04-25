"""
Evaluate fine-tuned Gemma 4 vs base model on packing list generation.

The fine-tuned model (base + LoRA adapter) is loaded ONCE.
For the base evaluation the LoRA adapter is disabled in-place — no second model load.

Usage:
    uv run main.py
    uv run main.py --num-samples 30 --output-dir results/run1
"""

import argparse
import os

import torch
from dotenv import load_dotenv
from unsloth import FastModel
from unsloth import get_chat_template

from gemma4_finetuned_packing_evaluation.config import (
    ADAPTER_ID,
    MAX_SEQ_LENGTH,
    NUM_SAMPLES,
)
from gemma4_finetuned_packing_evaluation.data import load_datasets, prepare_eval_data
from gemma4_finetuned_packing_evaluation.evaluator import SmartPackEvaluator
from gemma4_finetuned_packing_evaluation.reporting import (
    plot_radar_comparison,
    print_judge_comments,
    print_metrics_comparison,
    print_results_table,
    print_summary,
    save_results,
    _normalize_metrics,
)

import pandas as pd


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash-lite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hf_token     = os.environ["HF_TOKEN"]
    gemini_key   = os.environ.get("GEMINI_API_KEY", "")

    # ── 1. Load fine-tuned model (base weights + LoRA adapter) ──────────────
    print(f"\nLoading fine-tuned model from {ADAPTER_ID}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=ADAPTER_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Remove multimodal towers to free VRAM
    inner = getattr(model, "base_model", model)
    inner_model = getattr(inner, "model", inner)
    for tower in ("audio_tower", "vision_tower"):
        if hasattr(inner_model, tower):
            delattr(inner_model, tower)
            print(f"  Removed {tower}")

    tokenizer = get_chat_template(tokenizer, chat_template="gemma4")
    model = FastModel.for_inference(model)
    torch.cuda.empty_cache()
    print("Model ready.\n")

    # ── 2. Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset_train, dataset_test = load_datasets(hf_token)
    prompts, reference_texts, few_shot_pairs = prepare_eval_data(
        dataset_train, dataset_test,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"  {len(prompts)} test samples, {len(few_shot_pairs)} few-shot examples\n")

    # ── 3. Evaluate fine-tuned (LoRA enabled) ────────────────────────────────
    print("=" * 60)
    print("EVALUATION — Fine-tuned (LoRA enabled)")
    print("=" * 60)
    evaluator_ft = SmartPackEvaluator(
        model=model,
        tokenizer=tokenizer,
        label="Fine-tuned",
        gemini_api_key=gemini_key,
        gemini_model_id=args.gemini_model,
        few_shot_pairs=None,
    )
    results_ft = evaluator_ft.run(prompts, reference_texts)

    # ── 4. Disable LoRA → evaluate base model (no second load) ──────────────
    print("\n" + "=" * 60)
    print("EVALUATION — Base model (LoRA disabled)")
    print("=" * 60)
    model.disable_adapter_layers()
    torch.cuda.empty_cache()

    evaluator_base = SmartPackEvaluator(
        model=model,
        tokenizer=tokenizer,
        label="Base",
        gemini_api_key=gemini_key,
        gemini_model_id=args.gemini_model,
        few_shot_pairs=few_shot_pairs,
    )
    results_base = evaluator_base.run(prompts, reference_texts)
    model.enable_adapter_layers()

    # ── 5. Results ───────────────────────────────────────────────────────────
    combined = pd.concat([results_ft, results_base], ignore_index=True)

    print_results_table(results_ft)
    print_results_table(results_base)
    print_judge_comments(combined)
    print_summary(combined)

    global_max_tps = max(results_ft["tps"].max(), results_base["tps"].max(), 1.0)
    metrics_ft   = _normalize_metrics(results_ft,   global_max_tps)
    metrics_base = _normalize_metrics(results_base, global_max_tps)
    print_metrics_comparison({"Fine-tuned": metrics_ft, "Base": metrics_base})

    fig = plot_radar_comparison(
        {"Fine-tuned": metrics_ft, "Base": metrics_base},
        title="SmartPack AI — Fine-tuned vs Base",
    )
    fig.write_html(f"{args.output_dir}/radar.html")

    save_results(combined, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
