import torch
import pandas as pd
import os
from torch.utils.data import Dataset



class BioactivityData(Dataset):
    def __init__(self, raw_data_path, processed_data_path, feature_selector=None):

        # if not os.path.exists(processed_data_path):
        #     feature_selector(raw_data_path, processed_data_path)
        # self.dataframe = pd.read_csv(processed_data_path)
        self.dataframe = pd.read_csv(raw_data_path).to_numpy()
        return
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample_row = torch.Tensor(self.dataframe[idx])
        x, y = sample_row[:-1], sample_row[-1:]
        return x, y