import torch
import pandas as pd
import os
from torch.utils.data import Dataset

class BioactivityData(Dataset):
    def __init__(self, features, labels, weights):

        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
        self.weights = None if weights is None else torch.Tensor(weights)
        return
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.weights is None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], self.labels[idx], self.weights[idx]


def get_data(data_path, feature_selector):
    # TODO: make sure header options, etc work here
    df = pd.read_csv(data_path, index_col=0)

    features = df.to_numpy()[:,:-3]
    training_mask = df["is_training"].to_numpy().astype('bool')

    if feature_selector is not None:
        features = feature_selector.transform(features)

    train_data = BioactivityData(features[training_mask],
                                 df["label"][training_mask].to_numpy(),
                                 None)

    valid_data = BioactivityData(features[~training_mask], 
                                 df["label"][~training_mask].to_numpy(),
                                 df["weight"][~training_mask].to_numpy())

    return train_data, valid_data