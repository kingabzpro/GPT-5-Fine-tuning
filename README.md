# GPT-5 GoEmotions Fine-tuning (OpenAI Python SDK + Harmony prompts)

This project fine-tunes `gpt-5-2025-08-07` on the GoEmotions dataset to build a
multi-label emotion classifier that returns strict JSON.

## Setup

1) Python 3.10+
2) Set `OPENAI_API_KEY`
3) Install deps:
   - `pip install -r requirements.txt`

## Data prep

- Creates `ft_data/train.jsonl`, `ft_data/valid.jsonl`, and `ft_data/labels.json`
- Each record contains Harmony-style `system` and `developer` messages and a
  user message `Comment:\n...`, plus the gold assistant JSON.

Run:
```
python scripts/prepare_go_emotions.py
```

## Fine-tune

Start a job (defaults are sane; see `--help` for options):
```
python scripts/create_ft_job.py
```

The script writes artifacts to `artifacts/latest.json` including the
`fine_tuned_model` ID.

Optional: watch an existing job’s live events:
```
python scripts/watch_job.py --job-id ftjob_XXXXXXXX
```

## Inference

Use Responses API with enforced JSON:
```
python scripts/infer.py --text "I love this so much, thank you!"
```

To specify a model:
```
python scripts/infer.py --model ft:gpt-5-2025-08-07:ORG:go-emotions:xxxx \
  --text "I'm so frustrated by this update"
```

## Evaluation (sampled)

Runs micro/macro F1 on a sample of the test set:
```
python scripts/eval_sample.py --samples 200 --seed 7
```

## Notes

- Base model: `gpt-5-2025-08-07`
- Dataset: `google-research-datasets/go_emotions` (simplified)
- Output is strict JSON: `{"labels": ["...", ...]}`
- Uses Harmony-style prompts in fine-tuning and inference for consistency
```