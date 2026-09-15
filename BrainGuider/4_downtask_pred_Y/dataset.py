import os
import torch
from torch.utils.data import Dataset, DataLoader

class fmri_Dataset(Dataset):
    def __init__(self, data_dict):
        self.X_time = data_dict["X_time"]
        self.Y_time = data_dict["Y_time"]

    def __len__(self):
        return self.X_time.shape[0]

    def __getitem__(self, idx):
        X = self.X_time[idx]
        Y = self.Y_time[idx]
        return X, Y
    

def get_dataloader(data_dir, batch_size, seed):
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(data_dir, "test.pt")

    data_train = torch.load(train_path, weights_only=True)
    data_val = torch.load(val_path, weights_only=True)
    data_test = torch.load(test_path, weights_only=True)

    dataset_train = fmri_Dataset(data_train)
    dataset_val = fmri_Dataset(data_val)
    dataset_test = fmri_Dataset(data_test)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)

    return dataloader_train, dataloader_val, dataloader_test



