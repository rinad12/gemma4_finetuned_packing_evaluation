import pandas as pd
from google import genai
from tqdm import tqdm

from .inference import generate_for_eval
from .metrics import (
    PackingItem,
    compute_bert_score,
    compute_expert_recall,
    infer_trip_type,
    judge_with_gemini,
    parse_model_output,
)


class SmartPackEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        label: str = "model",
        gemini_api_key: str = "",
        gemini_model_id: str = "gemini-2.0-flash-lite",
        few_shot_pairs: list[tuple[str, str]] | None = None,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.label          = label
        self.few_shot_pairs = few_shot_pairs
        self._gemini        = None
        self._gm_id         = gemini_model_id

        if gemini_api_key:
            self._gemini = genai.Client(api_key=gemini_api_key)
            print(f"[{label}] Gemini judge: {gemini_model_id}")
        else:
            print(f"[{label}] Gemini API key not set — LLM judge will be skipped.")

        mode = f"few-shot ({len(few_shot_pairs)} examples)" if few_shot_pairs else "zero-shot"
        print(f"[{label}] Inference mode: {mode}")

    def _eval_single(self, prompt: str, reference_text: str = "") -> dict:
        trip_type = infer_trip_type(prompt)
        raw_text, tps, n_tokens = generate_for_eval(
            self.model, self.tokenizer, prompt, few_shot_pairs=self.few_shot_pairs
        )

        items, parse_error = parse_model_output(raw_text)
        json_valid    = bool(items) and (parse_error is None)
        expert_recall = compute_expert_recall(items, trip_type) if json_valid else 0.0
        bert_f1       = compute_bert_score(items, reference_text) if json_valid and reference_text else 0.0

        judge = {"accuracy": None, "expertise": None, "logic": None, "comment": "skipped"}
        if self._gemini and json_valid:
            judge = judge_with_gemini(self._gemini, prompt, items, self._gm_id)

        return {
            "model":         self.label,
            "prompt":        prompt,
            "trip_type":     trip_type,
            "n_items":       len(items),
            "n_new_tokens":  n_tokens,
            "tps":           tps,
            "json_valid":    json_valid,
            "parse_error":   parse_error or "",
            "expert_recall": expert_recall,
            "bert_score_f1": bert_f1,
            "accuracy":      judge.get("accuracy"),
            "expertise":     judge.get("expertise"),
            "logic":         judge.get("logic"),
            "judge_comment": judge.get("comment", ""),
            "raw_output":    raw_text,
        }

    def run(self, prompts: list[str], reference_texts: list[str] | None = None) -> pd.DataFrame:
        if reference_texts is None:
            reference_texts = [""] * len(prompts)
        rows = [
            self._eval_single(prompt, ref)
            for prompt, ref in tqdm(zip(prompts, reference_texts), total=len(prompts), desc=self.label)
        ]
        return pd.DataFrame(rows)
