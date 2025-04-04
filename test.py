import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from util import Dataset
from model.noisa import Noisa
import pytorch_lightning as pl
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description="Test GPT model perplexity with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=r'config/noisa-tiny.yaml', help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenization length")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--model_path", type=str, default=r'checkpoints/noisa-epoch=305-train_loss=0.01.ckpt', help="Path to the trained model checkpoint")
    return parser.parse_args()

def load_config(args):
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args

class PerplexityModule(pl.LightningModule):
    def __init__(self, model, max_length=1024):
        super().__init__()
        self.model = model
        self.model.register_buffer("attention_mask", torch.triu(torch.ones(max_length, max_length), diagonal=1).bool())

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = self.model.attention_mask
        logits = self(input_ids, attention_mask)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
        loss = loss.view(input_ids.size(0), -1).mean(dim=1)
        perplexity = torch.exp(loss).mean()
        self.log("perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True)
        return perplexity

def main():
    args = get_args()
    args = load_config(args)

    processed_dataset = load_from_disk("./data/humaneval")
    test_dataset = Dataset(processed_dataset, split='train')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = Noisa(embed_dim=args.embed_dim, num_heads=args.num_heads).to('cuda')
    testing_module = PerplexityModule(model=model, max_length=args.max_length)

    print("Loading model from checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cuda')

    # Load the model using the LightningModule to match the saved state dict
    testing_module.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully.")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
    )

    print("Starting perplexity evaluation...")
    trainer.test(testing_module, test_dataloader)

if __name__ == "__main__":
    main()
