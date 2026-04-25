import json
import re
import time
import warnings
from typing import Optional, Union

from bert_score import score as _bert_score_fn
from google import genai
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------- Packing item schema ----------

class PackingItem(BaseModel):
    item:     str             = Field(..., min_length=1)
    quantity: Union[int, str] = Field(...)
    reason:   str             = Field(..., min_length=1)

    @field_validator("quantity", mode="before")
    @classmethod
    def coerce_quantity(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"quantity must be numeric, got: {v!r}")

    @field_validator("item", "reason", mode="before")
    @classmethod
    def strip_ws(cls, v):
        return v.strip() if isinstance(v, str) else v


def _normalize_keys(obj: dict) -> dict:
    return {k.strip().lower(): v for k, v in obj.items()}


def _flatten_packing_list(raw_list: list) -> list[dict]:
    flat = []
    for entry in raw_list:
        if isinstance(entry, dict):
            normalized = _normalize_keys(entry)
            if "items" in normalized and isinstance(normalized["items"], list):
                flat.extend(normalized["items"])
            elif "item" in normalized:
                flat.append(normalized)
    return flat


def parse_model_output(raw_text: str) -> tuple[list[PackingItem], Optional[str]]:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    array_matches = list(re.finditer(r"\[[\s\S]*?\]", cleaned))
    if not array_matches:
        return [], "No JSON array found in output"

    for match in reversed(array_matches):
        try:
            raw_list = json.loads(match.group())
        except json.JSONDecodeError:
            continue

        flat = _flatten_packing_list(raw_list)
        if not flat:
            continue

        items, errors = [], []
        for idx, obj in enumerate(flat):
            try:
                items.append(PackingItem(**(_normalize_keys(obj) if isinstance(obj, dict) else obj)))
            except (ValidationError, TypeError) as exc:
                errors.append(f"item[{idx}]: {exc}")

        if items:
            return items, ("; ".join(errors) if errors else None)

    return [], "No valid PackingItem objects found in any JSON array"


# ---------- Expert recall ----------

EXPERT_KEYWORDS: dict[str, list[str]] = {
    "extreme":     ["emergency blanket", "first aid", "water purification",
                    "fire starter", "rope", "thermal", "ration"],
    "photography": ["camera", "tripod", "lens", "sd card", "power bank",
                    "battery", "filter", "cleaning kit"],
    "business":    ["laptop", "adapter", "power bank", "charger",
                    "documents", "suit", "business card"],
    "winter":      ["thermal", "insulated", "gloves", "wool",
                    "antifreeze", "boots", "hand warmer"],
    "monsoon":     ["waterproof", "rain jacket", "dry bag",
                    "moisture", "poncho", "leech sock"],
    "sports":      ["uniform", "first aid", "hydration", "compression",
                    "electrolyte", "tape", "muscle rub"],
    "default":     ["power bank", "first aid", "charger",
                    "documents", "water bottle"],
}

TRIP_TYPE_PATTERNS: dict[str, list[str]] = {
    "extreme":     ["extreme tourism", "expedition", "survival", "rescue"],
    "photography": ["photography", "photo shoot", "photographer"],
    "business":    ["business", "conference", "corporate", "summit"],
    "winter":      ["winter", "polar", "tundra", "subpolar", "arctic", "snow"],
    "monsoon":     ["monsoon", "heavy rain", "tropical", "humid"],
    "sports":      ["sports event", "competition", "tournament", "athlete"],
}


def infer_trip_type(prompt: str) -> str:
    text = prompt.lower()
    for trip_type, patterns in TRIP_TYPE_PATTERNS.items():
        if any(p in text for p in patterns):
            return trip_type
    return "default"


def compute_expert_recall(items: list[PackingItem], trip_type: str) -> float:
    keywords = EXPERT_KEYWORDS.get(trip_type, EXPERT_KEYWORDS["default"])
    full_text = " ".join(f"{it.item} {it.reason}".lower() for it in items)
    hits = sum(1 for kw in keywords if kw.lower() in full_text)
    return round(hits / len(keywords), 4) if keywords else 0.0


# ---------- BERTScore ----------

def compute_bert_score(items: list[PackingItem], reference_text: str, lang: str = "en") -> float:
    if not items or not reference_text.strip():
        return 0.0
    candidates = [it.reason for it in items]
    references  = [reference_text] * len(candidates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, F1 = _bert_score_fn(candidates, references, lang=lang, verbose=False)
    return round(F1.mean().item(), 4)


# ---------- Gemini judge ----------

_JUDGE_PROMPT = """\
You are a senior travel logistics consultant. Evaluate the packing list below for the given trip.

Trip description:
{trip_description}

Packing list (JSON):
{packing_list_json}

Rate on three criteria, each from 1 (very poor) to 5 (excellent):
- Accuracy:  Are items correct and well-suited for this specific trip?
- Expertise: Does the list reflect expert knowledge (specialised gear, safety, layering)?
- Logic:     Is the reason for each item clear and internally consistent?

Respond ONLY with a single valid JSON object, no other text:
{{"accuracy": <int 1-5>, "expertise": <int 1-5>, "logic": <int 1-5>, "comment": "<one concise sentence>"}}
"""

_BACKOFF_DELAYS = [1, 2, 4, 8, 16]


def _call_gemini(client: genai.Client, model_id: str, prompt: str, max_attempts: int = 5) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model_id, contents=prompt).text
        except Exception as exc:
            if attempt < max_attempts:
                wait = _BACKOFF_DELAYS[attempt - 1]
                print(f"  [Gemini] attempt {attempt}/{max_attempts} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [Gemini] all {max_attempts} attempts failed: {exc}")
    return None


def judge_with_gemini(
    client: genai.Client,
    trip_description: str,
    items: list[PackingItem],
    model_id: str = "gemini-2.0-flash-lite",
) -> dict:
    packing_json = json.dumps([it.model_dump() for it in items], indent=2)
    prompt = _JUDGE_PROMPT.format(trip_description=trip_description, packing_list_json=packing_json)
    raw = _call_gemini(client, model_id, prompt)
    if raw is None:
        return {"accuracy": None, "expertise": None, "logic": None, "comment": "API call failed"}

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"accuracy": None, "expertise": None, "logic": None, "comment": f"Parse error: {raw[:80]}"}

    try:
        scores = json.loads(match.group())
        for key in ("accuracy", "expertise", "logic"):
            if isinstance(scores.get(key), (int, float)):
                scores[key] = max(1, min(5, int(scores[key])))
        return scores
    except json.JSONDecodeError as exc:
        return {"accuracy": None, "expertise": None, "logic": None, "comment": f"JSONDecodeError: {exc}"}
