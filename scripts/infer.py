# Runs inference against your fine-tuned model with strict JSON output.

import argparse
import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from common import (
    load_latest_model_id,
    load_labels,
    build_developer_content,
    HARMONY_SYSTEM,
)

load_dotenv()


def run_inference(
    client: OpenAI,
    model: str,
    text: str,
    labels: List[str],
    max_output_tokens: int = 64,
) -> dict:
    messages = [
        {"role": "system", "content": HARMONY_SYSTEM},
        {"role": "developer", "content": build_developer_content(labels)},
        {"role": "user", "content": f"Comment:\n{text}"},
    ]
    resp = client.responses.create(
        model=model,
        input=messages,
        response_format={"type": "json_object"},
        max_output_tokens=max_output_tokens,
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"labels": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="FT model ID; defaults to artifacts/latest.json",
    )
    parser.add_argument("--text", required=True, help="Input comment text")
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in env"
    model: Optional[str] = args.model or load_latest_model_id()
    if not model:
        raise SystemExit("No model provided and artifacts/latest.json not found")

    labels = load_labels()
    client = OpenAI()
    out = run_inference(client, model, args.text, labels)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
