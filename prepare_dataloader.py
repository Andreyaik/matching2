import torch
from torch.utils.data import DataLoader


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_tensor):
        self.encodings_tensor = encodings_tensor

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings_tensor.items()}
        return item

    def __len__(self):
        return len(self.encodings_tensor['input_ids'])


def dataloader_my(data, tokenizer, max_length=256, batch_size=32):
    data_tensor = tokenizer(data, padding=True, max_length=max_length,
                            truncation=True, return_tensors='pt')
    data_dataset = IMDbDataset(data_tensor)
    dataloader_data = DataLoader(data_dataset, batch_size=batch_size)

    return dataloader_data








