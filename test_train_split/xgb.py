'''

XGBoost for feature selection

This script is for feature extraction and the subsequent training 
of XGBoost for classification.

The routine trains the model twice: the first time extracts the most important features;
the second run trains the model for classification.

Inputs for the fit method:
  
  X: (array_like) – Feature matrix. Note only feature columns
  should be included (drop 'cid' and 'pid' etc.)

  Y: (array_like) – 'activity' values (0 or 1)

  sample_weight: (array_like) – instance weights


Note that the X set should be sampled according to sample_activity_score:

  X = df_features[df_features['sample_activity_score'] > 0.1].values

'''
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array, check_random_state

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class XGBoostClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, Y, sample_weight=None):
        '''
        Steps are:

        (1) train XGBoost on the training data
        (2) obtain the features that provide the most gain
        (3) train the final model on the reduced feature set
        '''
        xgb = self.__train(X, Y, sample_weight)
        self.selected_features = self.__select_features(xgb)
        X_reduced = X.T[self.selected_features].T

        self.model = self.__train(X_reduced, Y, sample_weight)

    def predict(self, X):
        X_reduced = X.T[self.selected_features].T
        return self.model.predict(X)

    def save(self, location):
        pickle_out = open(location,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, location):
        serialized_fs = open(location, 'rb')
        fs = pickle.load(serialized_fs)
        self.feature_importance = fs.feature_importance

    def __select_features(self, model):
        gain_importance = model.get_booster().get_score(importance_type="gain")
        features = [int(key.split('f')[1]) for key in gain_importance.keys()]
        return features

    def __train(self, X, Y, sample_weight=None):
        self.random_state_ = check_random_state(self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
          X, Y, test_size=0.2, random_state=self.random_state_)

        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        '''
        The following parameters were determined via previous
        experimentation using gridsearch.
        '''
        xgb = XGBClassifier(learning_rate=0.02, 
            n_estimators=600, 
            objective='binary:logistic',
            silent=True, nthread=1,
            subsample=0.6,
            min_child_weight=1,
            max_depth=6,
            gamma=5,
            colsample_bytree=0.8)

        xgb.fit(
          X_train, 
          y_train, 
          sample_weight=sample_weight,
          eval_metric=["error", "logloss"], 
          eval_set=eval_set, 
          verbose=False, 
          early_stopping_rounds=50)

        return xgb


if __name__== "__main__":
    N_SAMPLES = 1000
    xgb = XGBoostClassifier(random_state=42)
    X = np.random.sample(size=(N_SAMPLES, 10))
    X_test = np.random.sample(size=(5, 10))
    Y = np.random.choice(2, size=N_SAMPLES, p=(0.5, 0.5))
    xgb.fit(X, Y)
    print(xgb.predict(X_test))



