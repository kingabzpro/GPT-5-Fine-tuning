# Uploads data and creates a fine-tuning job on gpt-5-2025-08-07.
# Polls until completion and saves artifacts/latest.json with the FT model ID.

import argparse
import json
import os
import time
from typing import Optional
from openai import OpenAI

from common import (
    ensure_dirs,
    save_latest,
    BASE_MODEL,
    FT_DATA_DIR,
    ARTIFACTS_DIR,
)


def create_job(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: Optional[str],
    base_model: str,
    n_epochs: int,
    lr_mult: float,
    batch_size: str,
    suffix: str,
):
    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=training_file_id,
        validation_file=validation_file_id,
        suffix=suffix,
        hyperparameters={
            "n_epochs": n_epochs,
            "learning_rate_multiplier": lr_mult,
            "batch_size": batch_size,
        },
    )
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=f"{FT_DATA_DIR}/train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--valid",
        default=f"{FT_DATA_DIR}/valid.jsonl",
        help="Path to valid.jsonl",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--lr-mult",
        type=float,
        default=1.0,
        help="Learning rate multiplier",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help='Batch size or "auto"',
    )
    parser.add_argument(
        "--suffix",
        default="go-emotions",
        help="Suffix for the fine-tuned model name",
    )
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in env"
    ensure_dirs()
    client = OpenAI()

    # Upload files
    with open(args.train, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with open(args.valid, "rb") as f:
        valid_file = client.files.create(file=f, purpose="fine-tune")

    print("Uploaded files:", train_file.id, valid_file.id)

    job = create_job(
        client=client,
        training_file_id=train_file.id,
        validation_file_id=valid_file.id,
        base_model=args.base_model,
        n_epochs=args.epochs,
        lr_mult=args.lr_mult,
        batch_size=args.batch_size,
        suffix=args.suffix,
    )
    print("Job created:", job.id)

    # Persist job stub
    with open(f"{ARTIFACTS_DIR}/job_{job.id}.json", "w", encoding="utf-8") as f:
        json.dump(job.model_dump(), f, ensure_ascii=False, indent=2)

    # Poll until done
    status = None
    while True:
        j = client.fine_tuning.jobs.retrieve(job.id)
        if j.status != status:
            print("Status:", j.status)
            status = j.status
        if j.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(15)

    if j.status != "succeeded":
        raise SystemExit(f"Fine-tune did not succeed: {j.status}")

    print("Fine-tuned model:", j.fine_tuned_model)
    save_latest(
        {
            "job_id": j.id,
            "fine_tuned_model": j.fine_tuned_model,
            "base_model": args.base_model,
            "training_file_id": train_file.id,
            "validation_file_id": valid_file.id,
        }
    )


if __name__ == "__main__":
    main()
