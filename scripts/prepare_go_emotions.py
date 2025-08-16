# Creates OpenAI fine-tuning JSONL (messages format) for GoEmotions.
# Embeds Harmony-style system + developer messages in each record and writes:
# - ft_data/train.jsonl
# - ft_data/valid.jsonl
# - ft_data/labels.json

import json
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from common import (
    ensure_dirs,
    build_developer_content,
    HARMONY_SYSTEM,
    FT_DATA_DIR,
    LABELS_PATH,
)


def make_record(
    text: str, label_ids: List[int], label_names: List[str]
) -> Dict[str, Any]:
    labels = sorted({label_names[i] for i in label_ids})
    dev_msg = build_developer_content(label_names)
    assistant_json = json.dumps({"labels": labels}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": HARMONY_SYSTEM},
            {"role": "developer", "content": dev_msg},
            {"role": "user", "content": f"Comment:\n{text}"},
            {"role": "assistant", "content": assistant_json},
        ]
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_dirs()
    ds = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        trust_remote_code=False,
    )
    label_names = ds["train"].features["labels"].feature.names  # type: ignore

    # Save labels to reuse during inference
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(label_names), f, ensure_ascii=False, indent=2)

    def convert_split(
        split_name: str, limit: int | None = None
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ex in tqdm(ds[split_name], desc=f"Converting {split_name}"):
            rows.append(make_record(ex["text"], ex["labels"], label_names))
            if limit and len(rows) >= limit:
                break
        return rows

    train_rows = convert_split("train", limit=500)
    valid_rows = convert_split("validation", limit=50)

    write_jsonl(f"{FT_DATA_DIR}/train.jsonl", train_rows)
    write_jsonl(f"{FT_DATA_DIR}/valid.jsonl", valid_rows)

    print("Wrote ft_data/train.jsonl, ft_data/valid.jsonl, ft_data/labels.json")


if __name__ == "__main__":
    main()
