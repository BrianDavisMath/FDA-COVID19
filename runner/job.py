'''

XGBoost for feature selection and modeling drug/protein binding

This script is for feature extraction and the subsequent training 
of XGBoost for classification, i.e. whether there will be a 
drug-receptor interactios.

The routine trains the model twice: the first time extracts the most 
important features; the second run trains the model for the lassification.

The dimension reduction and modeling stage are repeated multiple times.
Each run is for a different subset of the training data that is determined
by an activity threshold. [Brian to document how that is calculated]

Inputs are:
  
  training_features.h5 - training feature matrix generated using 
  notebooks/greg-features.ipynb. This also includes the activity score
  that is used to sample a sub-training set for each run.

  validation_features.h5 - validation feature matrix generated using 
  notebooks/greg-features.ipynb

  max_activity_threshold - used to sub-sample the training data for the
  first run

  activity_threshold_step - the amount to reduced the activity threshold
  on each run before sub-sampling the training data


Outputs:

  results.csv - model outputs including binary classification for activity 
  as well as probability of activity for each cid/pid pair in the
  validation set;

  metrics.csv - the accuracy, precision, recall, weighted and unweighted
  F1 score representing model performane

'''
import sys, getopt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

class XGBoostClassifier():
    
    def __init__(
      self, 
      max_activity_threshold, 
      activity_threshold_step,
      training_features,
      validation_features):
        self.max_activity_threshold = max_activity_threshold
        self.activity_threshold_step = activity_threshold_step
        self.training_features = training_features
        self.validation_features = validation_features


def main(argv):
  training_features = None
  validation_features = None
  max_activity_threshold = None
  activity_threshold_step = None
  try:
    opts, args = getopt.getopt(argv,"ht:v:a:s:",["tfile=","vfile=", "athresh=", "step="])
  except getopt.GetoptError:
    print('job.py -t <training_features> -v <validation_features> -a \
<max_activity_threshold> -s <activity_threshold_step>')
    sys.exit(2)

  if len(opts) != 4:
    print('job.py -t <training_features> -v <validation_features> -a \
<max_activity_threshold> -s <activity_threshold_step>')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('job.py -t <training_features> -v <validation_features> -a \
<max_activity_threshold> -s <activity_threshold_step>')
      sys.exit()
    elif opt in ("-t", "--tfile"):
      training_features = arg
    elif opt in ("-v", "--vfile"):
      validation_features = arg
    elif opt in ("-a", "--athresh"):
      max_activity_threshold = arg
    elif opt in ("-s", "--step"):
      activity_threshold_step = arg

  if training_features is None or validation_features is None \
  or max_activity_threshold is None or activity_threshold_step is None:
    print('job.py -t <training_features> -v <validation_features> -a \
<max_activity_threshold> -s <activity_threshold_step>')
    sys.exit()

  print('training_features file is {}'.format(training_features))
  print('validation_features file is {}'.format(validation_features))
  print('max_activity_threshold is {}'.format(max_activity_threshold))
  print('activity_threshold_step is {}'.format(activity_threshold_step))


if __name__== "__main__":
  main(sys.argv[1:])




