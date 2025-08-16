# Streams fine-tuning events for an existing job ID.

import argparse
import os
import time

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Fine-tune job ID")
    parser.add_argument("--poll", type=float, default=10.0, help="Poll secs")
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in env"
    client = OpenAI()

    seen = set()
    while True:
        events = client.fine_tuning.jobs.list_events(args.job_id)
        for ev in events.data[::-1]:
            if ev.id in seen:
                continue
            seen.add(ev.id)
            ts = getattr(ev, "created_at", None)
            lvl = getattr(ev, "level", "info")
            msg = getattr(ev, "message", "")
            print(f"[{lvl}] {msg}")
        job = client.fine_tuning.jobs.retrieve(args.job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            print("Job status:", job.status)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
