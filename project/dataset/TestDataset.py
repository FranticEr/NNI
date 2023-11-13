#产生不同形状的仿真dataset
from torch.utils.data import Dataset
import torch
from typing import Any



class TestDataset(Dataset):
    def __init__(self,shape:list):
        self.TestData=torch.randn(shape)
    def __len__(self):
        return len
    def __getitem__(self, index) -> Any:
        return self.TestData[index]