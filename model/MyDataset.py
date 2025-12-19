import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

class MyTorchDataset(Dataset):
    def __init__(self, label_with_timestamp, channels, aj_matrix, window_size):
        self.time_list = self.read_label(label_with_timestamp)
        # ensure numeric arrays (float for channels, int for timestamps)
        self.channels = channels.astype(float)
        self.len = len(self.time_list)
        self.aj_matrix = aj_matrix
        self.window_size = window_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, i):
        next_channel = self.channels[:, i+self.window_size:i+self.window_size+1].astype(float)
        next_timestamp = self.time_list[i+self.window_size:i+self.window_size+1].astype('int64')
        aj = self.aj_matrix.astype(float)
        this_channel = self.channels[:, i:i+self.window_size].astype(float)
        return (
            torch.from_numpy(next_channel),
            torch.from_numpy(aj),
            torch.from_numpy(this_channel),
            torch.from_numpy(next_timestamp),
        )
        
    def __len__(self):
        # Ensure non-negative length to satisfy DataLoader requirements
        return max(0, self.len - self.window_size - 1)
    
    def read_label(self, label_with_timestasmp):
        return pd.to_numeric(label_with_timestasmp['timestamp'], errors='coerce').fillna(0).astype('int64').values
