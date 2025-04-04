import argparse
import yaml
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess HumanEval dataset with custom tokenizer settings")
    parser.add_argument("--config", type=str, default="config/noisa-tiny.yaml", help="Path to YAML config file")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenization length")
    return parser.parse_args()

def load_config(args):
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args

def main():
    args = get_args()
    args = load_config(args)
    max_length = args.max_length

    os.makedirs("./huggingface", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    dataset = load_dataset("alvations/c4p0", cache_dir="./huggingface")
    # dataset = load_dataset("openai_humaneval", cache_dir="./huggingface")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./huggingface")
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    tokenizer.add_special_tokens({'bos_token': '<|bos|>'})
    tokenizer.add_special_tokens({'eos_token': '<|eos|>'})

    def format_text(example):
        return {
            # "text": (
            #     f"<|bos|>{example['prompt']}"
            #     f"\ncode:\n{example['canonical_solution']}"
            #     f"\nentry point:\n{example['entry_point']}"
            #     f"\ntest:\n{example['test']}<|eos|>"
            # )
            "text": (
                f"<|bos|>{example['source']}<|eos|>"
            )
        }

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Adding attention_mask explicitly to the tokenized output
        return tokenized

    processed_dataset = dataset.map(format_text).map(
        tokenize_function,
        batched=True,
        # remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"]
    )

    # split_dataset = processed_dataset["test"].train_test_split(test_size=0.1, seed=42)
    split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=42)
    split_dataset.save_to_disk("./data/c4p0")
    tokenizer.save_pretrained("./huggingface/tokenizer")

    print("Tokenizer vocab size:", len(tokenizer))
    print("Finish tokenize and save preprocessed dataset.")

if __name__ == "__main__":
    main()
