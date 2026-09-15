import os
import torch
from torch.utils.data import Dataset, DataLoader

class fmri_Dataset(Dataset):
    def __init__(self, data_dict_s100):
        self.X_time = data_dict_s100["X_time"]
        
    def __len__(self):
        return self.X_time.shape[0]

    def __getitem__(self, idx):
        X = self.X_time[idx]
        return X


def get_dataloader(data_dir_s100, batch_size, seed):

    train_path_s100 = os.path.join(data_dir_s100, "train.pt")
    val_path_s100 = os.path.join(data_dir_s100, "val.pt")
    test_path_s100 = os.path.join(data_dir_s100, "test.pt")

    data_train_s100 = torch.load(train_path_s100, weights_only=True)
    data_val_s100 = torch.load(val_path_s100, weights_only=True)
    data_test_s100 = torch.load(test_path_s100, weights_only=True)

    dataset_train = fmri_Dataset(data_train_s100)
    dataset_val = fmri_Dataset(data_val_s100)
    dataset_test = fmri_Dataset(data_test_s100)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)

    return dataloader_train, dataloader_val, dataloader_test

