import argparse
import torch
from model.noisa import Noisa
import pytorch_lightning as pl
from transformers import AutoTokenizer
from train import TrainingModule
import yaml

def get_args():
    parser = argparse.ArgumentParser(description="Inference with trained Noisa model")
    parser.add_argument("--config", type=str, default=r'config/noisa-tiny.yaml', help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, default=r'./checkpoints/noisa-epoch=1341-train_loss=0.00.ckpt', help="Path to the trained model checkpoint")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenization length")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference")
    return parser.parse_args()


def load_config(args):
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args


def load_model(args):
    model = Noisa(embed_dim=args.embed_dim, num_heads=args.num_heads)
    training_module = TrainingModule.load_from_checkpoint(
        args.model_path,
        model=model,
        lr=args.lr,
        max_length=args.max_length
    )
    model = training_module.model
    model = model.to(args.device)
    model.eval()
    return model


def stream_inference(model, tokenizer, prompt, max_length, temperature=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    eos_token_id = tokenizer.eos_token_id
    print('Tokenizer length: ',len(tokenizer))
    with torch.no_grad():
        for _ in range(max_length):
            seq_len = input_ids.size(1)
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(input_ids.device)
            logits = model(input_ids, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
            print(generated_token, end="", flush=True)
            
            if next_token.item() == eos_token_id:
                break
    print()

def main():
    args = get_args()
    args = load_config(args)
    tokenizer = AutoTokenizer.from_pretrained("./huggingface/tokenizer")
    model = load_model(args)
    while True:
        prompt = input("Enter your prompt: ")
        stream_inference(model, tokenizer, prompt, args.max_length)


if __name__ == "__main__":
    main()
