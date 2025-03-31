import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from util import Dataset
from model.noisa import Noisa
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch.nn.functional as F
import os


def get_args():
    parser = argparse.ArgumentParser(description="Train GPT model with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=r'config/noisa-tiny.yaml', help="Path to YAML config file")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenization length")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention head")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--resume", type=str, default=r'', help="Path to checkpoint to resume training from")
    return parser.parse_args()

def load_config(args):
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args

class TrainingModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4, max_length=1024):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model.register_buffer("attention_mask", torch.triu(torch.ones(max_length, max_length), diagonal=1).bool())

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = self.model.attention_mask
        logits = self(input_ids, attention_mask)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    args = get_args()
    args = load_config(args)

    processed_dataset = load_from_disk("./data/humaneval")
    train_dataset = Dataset(processed_dataset, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    model = Noisa(embed_dim=args.embed_dim, num_heads=args.num_heads).to('cuda')
    print("Model architecture:")
    print(model)

    training_module = TrainingModule(model=model, lr=args.lr, max_length=args.max_length)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="noisa-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=False,
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=False
    )
    if args.resume:
        trainer.fit(training_module, train_dataloader, ckpt_path=args.resume)
    else:
        trainer.fit(training_module, train_dataloader)

if __name__ == "__main__":
    main()