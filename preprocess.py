from datasets import load_dataset
from transformers import AutoTokenizer
import os

os.makedirs("./huggingface", exist_ok=True)
os.makedirs("./data", exist_ok=True)

dataset = load_dataset("openai_humaneval", cache_dir="./huggingface")
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./huggingface")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def format_text(example):
    return {
        "text": (
            f"<input>{example['prompt']}</input>"
            f"<output>{example['canonical_solution']}</output>"
            f"<entry_point>{example['entry_point']}</entry_point>"
            f"<test>{example['test']}</test>"
        )
    }

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )

processed_dataset = dataset.map(format_text).map(
    tokenize_function,
    batched=True,
    remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"]
)

split_dataset = processed_dataset["test"].train_test_split(test_size=0.1)

split_dataset.save_to_disk("./data/humaneval")
tokenizer.save_pretrained("./huggingface/tokenizer")

print("Tokenizer vocab size: ", len(tokenizer))
print("HumanEval preprocessing complete! Files saved to:")
print("- Processed data: ./data/humaneval")
print("- Custom tokenizer: ./huggingface/tokenizer")
