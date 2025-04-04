import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split='train', columns=['input_ids', 'attention_mask']):
        self.dataset = dataset[split]
        self.dataset.set_format(type='torch', columns=columns)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
