from warnings import warn
import warnings
import numpy as np
import pickle
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from deap import base
from deap import creator
from deap import tools
from abc import ABC, abstractmethod

class BaseFeatureSelectionModel(ABC):
    @abstractmethod
    def fit(self, features, labels, num_trials=25, randseed=42):
        pass

    @abstractmethod
    def transform(self, features, num_keep_features):
        pass

    @abstractmethod
    def save(self, location):
        pass

    @abstractmethod
    def load(self, location):
        pass


class RandomForestFeatureSelection(BaseFeatureSelectionModel):
    def __init__(self):
        self.feature_importance = None

    def fit(self, features, labels, num_trials=25, randseed=42):
        np.random.seed(randseed)
        random_seeds = np.random.choice(num_trials ** 2, size=num_trials, replace=False).tolist()
        self.feature_importance = np.zeros(n_dims)
        for i in range(num_trials):
            rf = RandomForestClassifier(random_state=random_seeds.pop(), n_estimators=10)
            rf.fit(features, labels)
            self.feature_importance += rf.feature_importances_

    def transform(self, features, num_keep_features):
        return np.take(features, np.argsort(self.feature_importance)[:num_keep_features], axis=1)

    def save(self, location, serialized_feature_selector=None):
        pickle_out = open(location,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, location):
        serialized_fs = open(location, 'rb')
        fs = pickle.load(serialized_fs)
        self.feature_importance = fs.feature_importance


# synthetic data
n_samples, n_dims = 1000, 100
input_array = np.random.sample(size=(n_samples, n_dims))
activity_labels = np.random.choice(2, size=n_samples, p=(0.95, 0.05))

# Random Forest Method: usage example
model_location = 'example_model'

feature_selector = RandomForestFeatureSelection()
feature_selector.fit(input_array, activity_labels)
feature_selector.save('example_model')

feature_selector2 = RandomForestFeatureSelection()
feature_selector2.load(model_location)
reduced_features = feature_selector2.transform(input_array, 10)
print(reduced_features)
