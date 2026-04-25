import json
import random

from datasets import load_dataset

from .config import DATASET_ID, NUM_FEW_SHOT, NUM_SAMPLES


def row_to_prompt(row: dict) -> str:
    risks = row["risks"]
    if isinstance(risks, list):
        risks = ", ".join(risks)
    fields = [
        "Trip Intent: " + str(row["intent"]),
        "Duration: " + str(row["duration"]) + " days",
        "Infrastructure: " + str(row["infrastructure"]),
        "Climate: " + str(row["climate"]),
        "Potential Risks: " + risks,
    ]
    return "\n".join(fields)


def row_to_output(row: dict) -> str:
    flat_list = []
    raw_list = row["packing_list"]
    if len(raw_list) > 0 and "items" in raw_list[0]:
        for category in raw_list:
            for item in category["items"]:
                flat_list.append({k: v for k, v in item.items() if k in ["item", "quantity", "reason"]})
    else:
        for item in raw_list:
            flat_list.append({k: v for k, v in item.items() if k in ["item", "quantity", "reason"]})

    return f"Reasoning: {row['reasoning']}\n\nPacking List: {json.dumps(flat_list, ensure_ascii=False)}"


def load_datasets(hf_token: str):
    dataset_test  = load_dataset(DATASET_ID, split="test",  token=hf_token)
    dataset_train = load_dataset(DATASET_ID, split="train", token=hf_token)
    return dataset_train, dataset_test


def prepare_eval_data(
    dataset_train,
    dataset_test,
    num_samples: int = NUM_SAMPLES,
    num_few_shot: int = NUM_FEW_SHOT,
    seed: int = 0,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    rng = random.Random(seed)
    test_indices = rng.sample(range(len(dataset_test)), num_samples)
    prompts = [row_to_prompt(dataset_test[i]) for i in test_indices]

    # Reference texts for BERTScore (ground-truth packing list reasons)
    reference_texts = []
    for i in test_indices:
        from .metrics import parse_model_output
        raw_gt = row_to_output(dataset_test[i])
        gt_items, _ = parse_model_output(raw_gt)
        reference_texts.append(" ".join(it.reason for it in gt_items) if gt_items else "")

    # Few-shot examples from train (fixed seed so they never overlap with test)
    fs_rng = random.Random(42)
    few_shot_rows = [dataset_train[i] for i in fs_rng.sample(range(len(dataset_train)), num_few_shot)]
    few_shot_pairs = [(row_to_prompt(r), row_to_output(r)) for r in few_shot_rows]

    return prompts, reference_texts, few_shot_pairs
