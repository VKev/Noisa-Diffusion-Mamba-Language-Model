import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from util import CustomDataset
from model.noisa import Noisa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_dataset = load_from_disk("./data/humaneval")
train_dataset = CustomDataset(processed_dataset, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = Noisa().to(device)
attention_mask = torch.triu(torch.ones(2048, 2048, device=device), diagonal=1).bool()

for batch in train_dataloader:
    input_ids = batch['input_ids'].to(device)
    x = model(input_ids, attention_mask)
    print(f"x Shape: {x.shape}")
    print(f"Input IDs Shape: {input_ids.shape}")
    break