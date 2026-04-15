"""
Convert conversations_gemma_finetune.jsonl → HuggingFace Dataset.

Outputs
-------
hf_dataset/              Arrow dataset (load with datasets.load_from_disk)
├── train/
└── validation/

If a Gemma tokenizer path is supplied via --tokenizer, an additional
"text" column is written with apply_chat_template output, ready for
SFTTrainer's dataset_text_field="text".

Usage
-----
# Minimal — just create the Arrow dataset
python convert_to_hf_dataset.py

# With chat-template formatting (requires tokenizer on disk or Hub ID)
python convert_to_hf_dataset.py --tokenizer google/gemma-4-9b-it

# Custom split ratio and output path
python convert_to_hf_dataset.py --val-ratio 0.05 --output my_dataset/
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", default="conversations_gemma_finetune.jsonl",
        help="Source JSONL file (default: conversations_gemma_finetune.jsonl)",
    )
    p.add_argument(
        "--output", default="hf_dataset",
        help="Output directory for Arrow dataset (default: hf_dataset/)",
    )
    p.add_argument(
        "--tokenizer", default=None,
        help="Gemma tokenizer path or Hub ID. When provided, adds a 'text' column "
             "via apply_chat_template for SFTTrainer.",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of data reserved for validation (default: 0.10)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)",
    )
    p.add_argument(
        "--push-to-hub", default=None,
        help="Optional Hub repo ID to push to, e.g. 'your-org/transplant-coordinator'",
    )
    return p.parse_args()


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── Schema normalisation ─────────────────────────────────────────────────────

KNOWN_SCENARIOS = [
    "elevated_creatinine",
    "followup_appointment",
    "emotional_distress",
    "lifestyle_changes",
    "borderline_creatinine",
    "tacrolimus_toxicity_vs_rejection",
    "stable_labs_concerning_symptoms",
    "longitudinal_stable_improving",
    "longitudinal_rocky_stabilizing",
    "longitudinal_concerning_progression",
    "longitudinal_relapse_recovery",
]


def normalise(record: dict) -> dict:
    """
    Flatten the record into HuggingFace-friendly columns.

    messages stays as a list-of-dicts with keys "role" and "content".
    Roles are kept as "system" / "user" / "model" — matching Gemma's
    chat template exactly. Do NOT rename "model" to "assistant" or
    apply_chat_template will produce wrong tokens.

    Metadata columns are cast to simple scalar types so Arrow can
    infer a consistent schema across all rows.
    """
    return {
        # ── identity ──
        "id": record["id"],
        # ── scenario metadata ──
        "scenario": record.get("scenario", ""),
        "transplant_type": record.get("transplant_type", ""),
        "months_post_transplant": int(record.get("months_post_transplant", 0)),
        # ── longitudinal metadata (empty string / 0 for non-longitudinal rows) ──
        "longitudinal_patient_id": record.get("longitudinal_patient_id", ""),
        "visit_number": int(record.get("visit_number", 0)),
        # ── conversation ──
        "messages": record["messages"],          # list[{role, content}]
        "num_turns": len(
            [m for m in record["messages"] if m["role"] == "model"]
        ),
    }


# ─── Optional: apply Gemma chat template ──────────────────────────────────────

def apply_template(dataset: Dataset, tokenizer_id: str) -> Dataset:
    """
    Add a "text" column by running tokenizer.apply_chat_template on each row.

    The resulting column is a single string in Gemma's native BOS/turn format,
    ready to pass to SFTTrainer(dataset_text_field="text").

    Requires: pip install transformers torch
    """
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def format_row(batch):
        texts = []
        for msgs in batch["messages"]:
            # apply_chat_template expects list[{role, content}]
            # add_generation_prompt=False → include the final model turn in the loss
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    print("Applying chat template…")
    return dataset.map(format_row, batched=True, batch_size=256)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load raw JSONL
    print(f"Loading {args.input}…")
    records = load_jsonl(args.input)
    print(f"  {len(records)} conversations loaded.")

    # 2. Normalise schema
    rows = [normalise(r) for r in records]

    # 3. Build HuggingFace Dataset
    dataset = Dataset.from_list(rows)
    print(f"  Schema: {dataset.features}")

    # 4. Apply chat template if tokenizer given
    if args.tokenizer:
        dataset = apply_template(dataset, args.tokenizer)

    # 5. Stratified train / validation split.
    #    datasets 2.20 + NumPy 2.x has a bug in stratify_by_column, so we
    #    stratify manually: shuffle each scenario bucket then take val_ratio
    #    from each, guaranteeing proportional representation.
    import random as _random
    from collections import defaultdict

    rng = _random.Random(args.seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[row["scenario"]].append(i)

    train_indices, val_indices = [], []
    for sc_indices in buckets.values():
        rng.shuffle(sc_indices)
        n_val = max(1, round(len(sc_indices) * args.val_ratio))
        val_indices.extend(sc_indices[:n_val])
        train_indices.extend(sc_indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    ds_dict = DatasetDict({
        "train": dataset.select(train_indices),
        "validation": dataset.select(val_indices),
    })

    print(f"\nSplit sizes:")
    print(f"  train:      {len(ds_dict['train']):>5}")
    print(f"  validation: {len(ds_dict['validation']):>5}")

    # 6. Save Arrow dataset
    out = Path(args.output)
    ds_dict.save_to_disk(str(out))
    print(f"\nSaved Arrow dataset → {out}/")
    print(f"  Load with: datasets.load_from_disk('{out}')")

    # 7. Optionally push to Hub
    if args.push_to_hub:
        print(f"\nPushing to Hub: {args.push_to_hub}")
        ds_dict.push_to_hub(args.push_to_hub)
        print("Done.")

    # 8. Print a sample to verify
    print("\n── Sample row (train[0]) ──")
    row = ds_dict["train"][0]
    print(f"  id:                     {row['id']}")
    print(f"  scenario:               {row['scenario']}")
    print(f"  transplant_type:        {row['transplant_type']}")
    print(f"  months_post_transplant: {row['months_post_transplant']}")
    print(f"  longitudinal_patient_id:{row['longitudinal_patient_id'] or '(none)'}")
    print(f"  visit_number:           {row['visit_number'] or '(none)'}")
    print(f"  num_turns:              {row['num_turns']}")
    print(f"  messages[0]['role']:    {row['messages'][0]['role']}")
    if "text" in row:
        print(f"  text[:200]:             {row['text'][:200]}")

    # 9. Scenario distribution
    from collections import Counter
    train_scenarios = Counter(ds_dict["train"]["scenario"])
    print("\n── Scenario distribution (train) ──")
    for sc, cnt in sorted(train_scenarios.items()):
        print(f"  {sc:<45} {cnt}")


if __name__ == "__main__":
    main()
