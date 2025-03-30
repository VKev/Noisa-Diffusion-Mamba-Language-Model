import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from util import Dataset
from model.noisa import Noisa

import pytorch_lightning as pl
import torch.nn.functional as F

class TrainingModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.register_buffer("attention_mask", torch.triu(torch.ones(2048, 2048), diagonal=1).bool())

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = self.attention_mask
        logits = self(input_ids, attention_mask)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

processed_dataset = load_from_disk("./data/humaneval")
train_dataset = Dataset(processed_dataset, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = Noisa()

training_module = TrainingModule(model=model, lr=1e-4)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None
)

trainer.fit(training_module, train_dataloader)
