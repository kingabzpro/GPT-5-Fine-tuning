# scripts/common.py
# Common constants and helpers reused across scripts.

import json
import os
from typing import Any, Dict, List, Optional

ARTIFACTS_DIR = "artifacts"
FT_DATA_DIR = "ft_data"
LABELS_PATH = os.path.join(FT_DATA_DIR, "labels.json")
LATEST_PATH = os.path.join(ARTIFACTS_DIR, "latest.json")

# Default base model (ensure your account has access and FT enabled)
BASE_MODEL = "gpt-5-mini"

# Harmony-style preamble content (kept constant for train/infer/eval)
HARMONY_SYSTEM = (
    "You are an emotion classifier.\n\n"
    "Knowledge cutoff: 2024-10\n"
    "Current date: 2025-08-16\n\n"
    "Reasoning: low"
)

# NOTE: Double braces {{ }} are required wherever we want literal { } in the
# output, because we call .format() on this template below.
HARMONY_DEV_TEMPLATE = (
    "# Task\n"
    "Classify the user's single comment using the GoEmotions label set. "
    "This is multi-label: return zero or more emotions.\n\n"
    "# Output format\n"
    "Respond only with a compact JSON object:\n"
    '{{"labels": ["label1", "label2", ...]}}\n\n'
    "# Rules\n"
    "- Use ONLY labels from the allowed set (exact strings).\n"
    "- You may return zero or more labels.\n"
    '- If none apply, return {{"labels": []}}.\n'
    "- Output must be valid minified JSON (no code fences, no prose).\n"
    "- Deduplicate labels; sort ascending alphabetically.\n\n"
    "# Allowed labels ({count})\n"
    "{allowed}"
)


def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(FT_DATA_DIR, exist_ok=True)


def build_developer_content(labels: List[str]) -> str:
    return HARMONY_DEV_TEMPLATE.format(
        count=len(labels), allowed=json.dumps(labels, ensure_ascii=False)
    )


def save_latest(payload: Dict[str, Any]) -> None:
    ensure_dirs()
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_latest_model_id() -> Optional[str]:
    if not os.path.exists(LATEST_PATH):
        return None
    try:
        with open(LATEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("fine_tuned_model")
    except Exception:
        return None


def load_labels() -> List[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError("labels.json is not a list")
    return [str(x) for x in labels]
