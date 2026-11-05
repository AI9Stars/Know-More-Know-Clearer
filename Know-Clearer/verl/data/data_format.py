"""Convert a QA JSON-Lines file into the parquet layout expected by verl.

Each input line must be a JSON object with a ``question`` and an ``answer`` field::

    {"question": "who discovered the sun was the center of the solar system",
     "answer": "Aristarchus of Samos"}

The output parquet carries ``data_source="qa-acc"`` so that
``verl/utils/reward_score/__init__.py`` routes it to ``qa_acc_reward``.

Usage::

    python data/data_format.py --input train.jsonl --output data/train-clean.parquet --split train
    python data/data_format.py --input dev.jsonl   --output data/dev-clean.parquet   --split test
"""

import argparse

from datasets import load_dataset

QA_Prompt = "Please think about the reasoning process in the mind and then provide the user with the answer directly. "
QA_template = 'Question: {}\nAnswer:'

DATA_SOURCE = 'qa-acc'


def format_data(data_path, data_type, split):
    dataset = load_dataset(data_type, data_files=data_path, split="train")
    dataset = dataset.add_column("id", range(len(dataset)))

    def map_func(example):
        messages = []
        id = example["id"]
        input = QA_Prompt + QA_template.format(example["question"])
        output = example["answer"]
        messages.append({"role": "user", "content": input})
        ability = 'QA'
        reward_model = {
            "ground_truth": output,
            "style": "rule"
        }
        extra_info = {
            "index": id,
            "split": split,
        }
        return {
            "data_source": DATA_SOURCE,
            "prompt": messages,
            'ability': ability,
            "reward_model": reward_model,
            "extra_info": extra_info,
        }

    dataset = dataset.map(map_func, batched=False)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Source jsonl file (one {question, answer} object per line).")
    parser.add_argument("--output", required=True, help="Destination parquet file.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Value written to extra_info.split.")
    args = parser.parse_args()

    data = format_data(args.input, 'json', args.split)
    data.to_parquet(args.output)
    print(f"wrote {len(data)} rows to {args.output} (data_source={DATA_SOURCE}, split={args.split})")


if __name__ == "__main__":
    main()
