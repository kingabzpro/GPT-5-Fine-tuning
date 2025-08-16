# Evaluates the fine-tuned model on a sampled test subset.
# Prints micro/macro F1. Keeps cost in check via --samples.

import argparse
import json
import os
import random
from typing import List, Set

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from common import (
    load_latest_model_id,
    build_developer_content,
    HARMONY_SYSTEM,
)

load_dotenv()


def f1_micro(preds: List[Set[str]], trues: List[Set[str]]) -> float:
    tp = fp = fn = 0
    for p, t in zip(preds, trues):
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def f1_macro(preds: List[Set[str]], trues: List[Set[str]], labels: List[str]) -> float:
    total = 0.0
    for lbl in labels:
        tp = fp = fn = 0
        for p, t in zip(preds, trues):
            in_p = lbl in p
            in_t = lbl in t
            if in_p and in_t:
                tp += 1
            elif in_p and not in_t:
                fp += 1
            elif (not in_p) and in_t:
                fn += 1
        denom = 2 * tp + fp + fn
        total += (2 * tp) / denom if denom > 0 else 0.0
    return total / len(labels) if labels else 0.0


def predict_labels(client: OpenAI, model: str, text: str, dev_msg: str) -> List[str]:
    messages = [
        {"role": "system", "content": HARMONY_SYSTEM},
        {"role": "developer", "content": dev_msg},
        {"role": "user", "content": f"Comment:\n{text}"},
    ]
    resp = client.responses.create(
        model=model,
        input=messages,
        response_format={"type": "json_object"},
        max_output_tokens=64,
    )
    try:
        data = json.loads(resp.output_text)
        if isinstance(data.get("labels"), list):
            return [str(x) for x in data["labels"]]
    except Exception:
        pass
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="FT model ID; defaults to artifacts/latest.json",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of test samples",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in env"
    model = args.model or load_latest_model_id()
    if not model:
        raise SystemExit("No model provided and artifacts/latest.json not found")

    ds = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        trust_remote_code=False,
    )
    label_names = ds["train"].features["labels"].feature.names  # type: ignore
    dev_msg = build_developer_content(label_names)

    test_rows = list(ds["test"])
    random.seed(args.seed)
    random.shuffle(test_rows)
    test_rows = test_rows[: args.samples]

    client = OpenAI()
    preds: List[Set[str]] = []
    golds: List[Set[str]] = []
    for ex in tqdm(test_rows, desc="Evaluating"):
        gold = {label_names[i] for i in ex["labels"]}
        pred = set(predict_labels(client, model, ex["text"], dev_msg))
        preds.append(pred)
        golds.append(gold)

    micro = f1_micro(preds, golds)
    macro = f1_macro(preds, golds, list(label_names))
    print(f"Sampled test micro-F1: {micro:.3f}")
    print(f"Sampled test macro-F1: {macro:.3f}")


if __name__ == "__main__":
    main()
