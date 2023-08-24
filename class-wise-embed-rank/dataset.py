import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class embed_rankDataset(Dataset):
    def __init__(self, base_dir):
        self.based_dir = base_dir
        self.labels = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
        self.list_dir = os.listdir(self.based_dir)

    def __len__(self):
        return len(self.based_dir)

    def getlabel(self):
        list_dir = os.listdir(self.based_dir)
