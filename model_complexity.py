import gzip
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances, jaccard_similarity_score, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import itertools
import pprint
import matplotlib.pyplot as plt
import csv


drug_features_file = '../FDA-COVID19_raw_data/dragon_features.csv'
protein_features_file = '../FDA-COVID19_raw_data/protein_features.csv'
interactions_file = '../FDA-COVID19_raw_data/interactions.txt'
drug_dict = {}
protein_dict = {}
features = []
labels = []

with open(drug_features_file, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csvreader):
        if i != 0:
            # assuming cid is first column, name is second column
            drug_dict[row[0]] = row[2:]

with open(protein_features_file, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csvreader):
        if i != 0:
            # assuming pid is second column
            protein_dict[row[1]] = row[2:]

with open(interactions_file, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csvreader):
        if i != 0:
            drug = drug_dict.get(row[1])
            protein = protein_dict.get(row[2])
            label = row[3]

            if drug is not None and protein is not None:
                combined_vector = (drug + protein)
                features.append(combined_vector)
                labels.append(label)

features = features[:50]
labels = labels[:50]

# removing columns with at least 5% "na"
num_na_list = []
for col_num, _ in enumerate(features[0]):
    num_na = 0
    for row_num, _ in enumerate(features):
        if features[row_num][col_num] == "na":
            num_na += 1
    num_na_list.append(num_na)

removed_cols = []
for i, num_na in enumerate(num_na_list):
    if num_na / len(features) >= 0.05:
        removed_cols.append(i)

features = np.array(features)
features = np.delete(features, removed_cols, axis=1)\

# removing remaining rows with anything that can't converted to a float
removed_rows = []
for i, _ in enumerate(features):
    for j, value in enumerate(features[i]):
        try:
            float_value = float(value)
        except:
            removed_rows.append(i)
            break
features = np.delete(features, removed_rows, axis=0)
labels = np.delete(labels, removed_rows, axis=0)

features = features.astype(np.float)
labels = labels.astype(np.float)

#fingerprints = np.vstack([active_prints, decoy_prints])
#labels = np.vstack([np.ones((len(active_prints), 1)), np.zeros((len(decoy_prints), 1))]).flatten().astype(int)

print("Training...")

print(features.shape)

outer_skf = StratifiedKFold(n_splits=5, shuffle=True)
random_splits = [(train, test) for train, test in outer_skf.split(features, labels)]
results_list = []
# for training_indices, test_indices in random_splits:

training_indices, test_indices = random_splits[0]
training_features = features[training_indices]
test_features = features[test_indices]
training_labels = labels[training_indices]
test_labels = labels[test_indices]

scores = []
grid_search = itertools.product(range(1, 200), range(1, 20))
for d, e in grid_search:
    model = RandomForestClassifier(n_estimators=d, max_depth=e, class_weight='balanced')
    model.fit(training_features, training_labels)
    predicted_probs = model.predict_proba(test_features)[:, 1]
    precision, recall, _ = precision_recall_curve(test_labels, predicted_probs)
    depth = max([estimator.tree_.max_depth for estimator in model.estimators_])
    scores.append([d, depth, np.round(auc(recall, precision), 5)])

print("DONE")
print(scores)

grid = np.array(scores)[:, :2]
perf = np.array(scores)[:, 2]
plt.scatter(grid[:,0], grid[:, 1], alpha=0.3, c=perf)
plt.xlabel('Number of trees')
plt.ylabel('Maximum tree depth')
plt.title('PR-AUC for random forest trained on random split: 11betaHSD1')
