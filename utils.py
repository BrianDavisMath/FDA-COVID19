import torch

import pandas as pd
import numpy as np
import time
import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--display-step", type=int, default=10)
    parser.add_argument("--fs", type=str, default="random_forest",
                        choices=["random_forest", "none"])
    parser.add_argument("--multigpu", action='store_true')

    return parser

# def get_feature_selector(selector_string):
#     if feature_selector == "random_forest":
#         feature_selector = RandomForestFeatureSelection()
#         feature_selector.load("location_of_saved_random_forest.pkl")
#     elif feature_selector == "sparse_pca":
#         feature_selector = SparsePCAFeatureSelection()

# def select_features(train_path, feature_selector, valid_path):
#     np_dataset = pd.read_csv(data_path).to_numpy()
#     x, y = np_dataset[:,:-1], np_dataset[:,-1]
#     selected_x = feature_selector.transform(x)

#     return selected_x, y