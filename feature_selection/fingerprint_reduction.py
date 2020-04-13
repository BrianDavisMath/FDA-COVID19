import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
data_loc = 'data/FDA-COVID19_files_v1.0/'
fda_cids = 'data/FDA-COVID19_files_v1.0/fda_drug_cids.csv'

def load_data(path, data_type=None):
    if data_type:
        df = pd.read_csv(path, index_col=0, dtype=data_type)
    else:
        df = pd.read_csv(path, index_col=0)
    return df


fda_cids = load_data(fda_cids)
df_fingerprints = load_data(data_loc+'drug_features/fingerprints.csv')
fda_fingerprints = df_fingerprints[df_fingerprints.index.astype(str).isin(fda_cids.cid)]


def merge_cols_at_pair(df, merge_pair):
    col1, col2 = merge_pair
    df.columns = df.columns.astype(str)
    df.loc[:, col1 + "_" + col2] = df.loc[:, [col1, col2]].apply(lambda row: int(any(row)), axis=1)
    return df


def reduce_columns(df, similarity_threshold=0.01, at_a_time=1):
    columns = list(df.columns)
    num_columns = len(columns)
    column_distances = pairwise_distances(df.values.transpose().astype(bool), metric="jaccard", n_jobs=-1)
    column_distances = 1 - column_distances * (1 - column_distances) - np.eye(num_columns)
    column_distances += similarity_threshold
    column_distances = column_distances.astype(int)
    if not np.any(column_distances):
        raise Exception("No collision-free merges possible.")
    else:
        sparsities = column_distances.sum(axis=0)
        merge_pairs = np.where(column_distances)
        num_pairs = len(merge_pairs[0])
        pair_scores = np.sqrt(np.product(np.take(sparsities, merge_pairs), axis=0))
        merge_pair_scores = np.argsort(pair_scores)[::-1]
        merge_pairs_ordered = np.vstack(merge_pairs)[:, merge_pair_scores]
        merged_cols = list()
        for i in range(min(at_a_time, num_pairs)):
            pair = merge_pairs_ordered[:, i].tolist()
            col_pair = [columns[index] for index in pair]
            if not any([col in merged_cols for col in col_pair]):
                df = merge_cols_at_pair(df, col_pair)
                merged_cols.extend(col_pair)
            else:
                pass
        df = df.drop(labels=merged_cols, axis=1)
        return df


reduced_df = deepcopy(fda_fingerprints)
total_var = reduced_df.var().sum()
_, num_cols = reduced_df.shape
print("numcols | total variance")
print(num_cols, total_var)
track_col_reduction = [num_cols]
track_var_reduction = [total_var]

# replace "while True" with "while num_cols < target_num_cols"
while True:
    reduced_df = reduce_columns(reduced_df, at_a_time=25)
    total_var = reduced_df.var().sum()
    track_var_reduction.append(total_var)
    _, num_cols = reduced_df.shape
    track_col_reduction.append(num_cols)
    print(num_cols, total_var)


plt.plot(track_col_reduction, [var / track_var_reduction[0] for var in track_var_reduction])
plt.xlabel("dimension")
plt.ylabel("percent retained variance")
plt.title("fingerprint dimension reduction")
