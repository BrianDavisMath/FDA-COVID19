'''

XGBoost for feature selection and modeling drug/protein binding

This script is for training XGBoost for classification, i.e. whether there will be a 
drug-receptor interactios. It carries out a Gridsearch to find the best parameters.


Inputs are:

  [-f] data_folder - defaults to "/data". Specifies the location of the data from which the
  features will be selected.

  [-n] name - job name used to place results in a directory of that name

  [-r] number of runs - used to determine the number of random samples to
  take from the hyperparameter space during fitting.

  [-c] generate continuous features files (one for cids and one for pids) for all interactions (non-zero for true)


Outputs:

  results.csv - target metric, accuracy, precision and recall along with the
  hyperparameters used to obtain the results

  Note: all results files are written to a dynamically created **results folder**.

  If the [-c] flag is set then the output will be a two h5 files containing all
  of the continuous features for cids and pids joined to the interactions file.


Data folder:

  The program expects a data folder containing subdirectories for drug and protein 
  features files as well as the split interactions files that are used to stitch 
  the features together into sets.

  The following files and folders are expected and should be included before 
  zipping up this for distribution:

    |____data
    | |____drug_features
    | | |____dragon_features.csv
    | | |____fingerprints.csv
    | |____protein_features
    | | |____binding_sites.csv
    | | |____profeat.csv
    |____coronavirus_features
    | | |____binding_sites.csv
    | | |____profeat.csv
    | |____training_validation_split
    | | |____weighted_interactions_v5.csv


Example call to run program:

  # train and gridsearch
  python job.py -f test_data/ -n 'test_0' -r 5

  # generate continuous feature files
  python job.py -f test_data/ -n 'test_0' -c 1


'''
import logging
import sys, getopt
import os
import gc
import random
import h5py
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logging.basicConfig(filename='job.log',level=logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logging.getLogger().addHandler(ch)


class XGBoostClassifier():
    
  def __init__(
    self, 
    data_folder,
    job_name,
    num_runs,
    generate_cont_features):

      self.data_loc = data_folder
      self.job_name = job_name
      self.num_runs = num_runs

      self.bad_dragon_cols = []

      # Create the feature sets for training and validation
      # Get the individual feature sets
      self.feature_sets = self.__load_feature_files()
      
      # load interactions.csv
      df_interactions = self.__load_data(
        self.data_loc+'training_validation_split/weighted_interactions_v5.csv')

      if generate_cont_features:
        self.__generate_continuous_features_file(df_interactions)
        return

      df_interactions['cid']=df_interactions['cid'].astype(int)

      df_training_interactions = df_interactions[
        (df_interactions['required_training']==True) | 
        (df_interactions['optional_training']==True)]

      df_validation_interactions = df_interactions[df_interactions['validation']==True]

      df_features = self.__create_features(self.feature_sets, df_training_interactions)
      df_validation = self.__create_features(self.feature_sets, df_validation_interactions)

      logging.debug('\nfeatures shape: {}'.format(df_features.shape))
      logging.debug('validation shape: {}\n'.format(df_validation.shape))

      del df_interactions
      del df_training_interactions
      del df_validation_interactions

      self.non_feature_columns = ['cid', 
        'pid', 
        'activity', 
        'required_training',
        'optional_training',
        'validation',
        'test',
        'prediction',
        'weight']
      
      # drop zero-variance columns
      var_cols = [col for col in df_features.columns if df_features[col].nunique() > 1]
      df = df_features[var_cols]

      logging.debug('Dropped {:,} columns that have zero variance.\n'.format(len(df_features.columns)-len(var_cols)))

      del df_features
      df_features = df

      logging.debug('Shape after dropping zero-variance columns - rows: {:,}, columns: {:,}\n'.
        format(len(df_features), len(df_features.columns)))

      # Run models
      l_learning_rate = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
      l_n_estimators = [10, 50, 100, 200, 500, 600]
      l_subsample = [0.1, 0.2, 0.4, 0.6, .08]
      l_min_child_weight = [1, 3, 5, 7 ]
      l_max_depth = [3, 4, 5, 6, 8, 10, 12, 15]
      l_gamma = [0.1, 0.5, 1.0, 2, 4, 6]
      l_colsample_bytree = [0.2, 0.4, 0.6, 0.8]
      l_sample_weights = [0, 1]

      results = []

      for i in range(self.num_runs):
        learning_rate = random.choice(l_learning_rate)
        n_estimators = random.choice(l_n_estimators)
        subsample = random.choice(l_subsample)
        min_child_weight = random.choice(l_min_child_weight)
        max_depth = random.choice(l_max_depth)
        gamma = random.choice(l_gamma)
        colsample_bytree = random.choice(l_colsample_bytree)
        use_sample_weights = random.choice(l_sample_weights)

        xgb = XGBClassifier(learning_rate=learning_rate, 
            n_estimators=n_estimators, 
            objective='binary:logistic',
            subsample=subsample,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            gamma=gamma,
            colsample_bytree=colsample_bytree)

        if use_sample_weights == 1:
          sample_weight = df_features['weight']
        else:
          sample_weight = None

        model_results = self.__train_and_eval(df_features, df_validation, xgb, sample_weight)
        gc.collect()

        target_metric = model_results['target_metric']
        accuracy = model_results['accuracy']
        precision = model_results['precision']
        recall = model_results['recall']

        results.append([
          target_metric,
          accuracy,
          precision,
          recall,
          learning_rate,
          n_estimators,
          subsample,
          min_child_weight,
          max_depth,
          gamma,
          colsample_bytree,
          use_sample_weights
        ])


      self.__results_to_csv(results)


  def __results_to_csv(self, results):
    df = pd.DataFrame(results, columns=[
      'target_metric',
      'accuracy',
      'precision',
      'recall',
      'learning_rate',
      'n_estimators',
      'subsample',
      'min_child_weight',
      'max_depth',
      'gamma',
      'colsample_bytree',
      'use_sample_weights'
    ])

    path = 'results/'+self.job_name
    try:
      os.makedirs(path, exist_ok=True)
    except OSError:
      logging.debug ("Creation of the directory %s failed" % path)
    else:
      logging.debug ("Successfully created the directory %s " % path)

    path = 'results/'+self.job_name+'/results.csv'
    df.to_csv(path, index=False)
    del df


  '''
  ==================================================================
  Misc functions
  ==================================================================
  '''
  def __create_cont_features(self, feature_sets, df_interactions):
    logging.debug('Creating continuous features files for cids and pids...\n')
    
    df_dragon_features = feature_sets['df_dragon_features']
    df_profeat = feature_sets['df_profeat']
    df_profeat_corona = feature_sets['df_profeat_corona']
    df_profeat_all = pd.concat([df_profeat, df_profeat_corona])
    
    logging.debug('df_profeat shape: {}'.format(df_profeat.shape))
    logging.debug('df_profeat_corona shape: {}'.format(df_profeat_corona.shape))
    logging.debug('df_profeat_all shape: {}\n'.format(df_profeat_all.shape))

    df_pid_features = pd.merge(df_interactions, df_profeat_all, on='pid', how='inner')
    
    df_dragon_features.index.name = 'cid'
    df_cid_features = pd.merge(df_interactions, df_dragon_features, on='cid', how='inner')
    
    # release memory used by previous dataframes.
    del df_profeat
    del df_profeat_corona
    del df_profeat_all
    del df_dragon_features
    
    return {'pid': df_pid_features, 'cid': df_cid_features}


  def __generate_continuous_features_file(self, df_interactions):
    '''
    Generate a features file containing all cid and pid continuous features
    joined to the interactions file.
    '''
    path = 'results/'+self.job_name
    try:
      os.makedirs(path, exist_ok=True)
    except OSError:
      logging.debug ("Creation of the directory %s failed" % path)
    else:
      logging.debug ("Successfully created the directory %s " % path)

    # Save continuous features to file.
    res = self.__create_cont_features(self.feature_sets, df_interactions)
    
    path = 'results/'+self.job_name+'/continuous_pid_features_v5.h5'
    store = pd.HDFStore(path)
    store['df'] = res['pid']
    store.close()

    path = 'results/'+self.job_name+'/continuous_cid_features_v5.h5'
    store = pd.HDFStore(path)
    store['df'] = res['cid']
    store.close()

    logging.debug ("Successfully saved continuous features to %s " % path)


  '''
  ==================================================================
  Functions for stitching together the individual features CSV files
  to create the training and validation sets.
  ==================================================================
  '''

  # load a specific features CSV file
  def __load_data(self, path, data_type=None):
    if data_type:
        df = pd.read_csv(path, index_col=0, dtype=data_type)
    else:
        df = pd.read_csv(path, index_col=0)
    
    columns_missing_values = df.columns[df.isnull().any()].tolist()
    
    return df


  # Get the individual feature sets as data frames
  def __load_feature_files(self):
    # note need to set the data_type to object because it complains, otherwise that the types vary.
    df_dragon_features = self.__load_data(self.data_loc+'drug_features/dragon_features.csv', data_type=object)
    
    # rename the dragon features since there are duplicate column names in the protein binding-sites data.
    df_dragon_features.columns = ['cid_'+col for col in df_dragon_features.columns]
    
    # handle na values in dragon_features
    # Many cells contain "na" values. Find the columns that contain 2% or 
    # less of these values and retain them, throwing away the rest. 
    # Then mean-impute the "na" values in the remaining columns.
    pct_threshold = 2
    na_threshold = int(len(df_dragon_features)*pct_threshold/100)
    ok_cols = []

    for col in df_dragon_features:
        na_count = df_dragon_features[col].value_counts().get('na')
        if (na_count or 0) <= na_threshold:
            ok_cols.append(col)
        else:
            self.bad_dragon_cols.append(col)

    logging.debug('number of columns where the frequency of "na" values is <= {}%: {}.'.format(pct_threshold, len(ok_cols)))
    
    df_dragon_features = df_dragon_features[ok_cols]

    # convert all values except "na"s to numbers and set "na" values to NaNs.
    df_dragon_features = df_dragon_features.apply(pd.to_numeric, errors='coerce')

    columns_missing_values = df_dragon_features.columns[df_dragon_features.isnull().any()].tolist()

    # replace NaNs with column means
    df_dragon_features.fillna(df_dragon_features.mean(), inplace=True)

    columns_missing_values = df_dragon_features.columns[df_dragon_features.isnull().any()].tolist()
    df_fingerprints = self.__load_data(self.data_loc+'drug_features/fingerprints.csv')
    
    df_binding_sites = self.__load_data(self.data_loc+'protein_features/binding_sites.csv')
    df_binding_sites_corona = self.__load_data(self.data_loc+'coronavirus_features/binding_sites.csv')
    
    # Name the index to 'pid' to allow joining to other feaure files later.
    df_binding_sites.index.name = 'pid'
    df_binding_sites_corona.index.name = 'pid'
    
    df_profeat = self.__load_data(self.data_loc+'protein_features/profeat.csv')
    df_profeat_corona = self.__load_data(self.data_loc+'coronavirus_features/profeat.csv')

    # Name the index to 'pid' to allow joining to other feaure files later.
    df_profeat.index.name = 'pid'
    df_profeat_corona.index.name = 'pid'
    
    # profeat has some missing values.
    s = df_profeat.isnull().sum(axis = 0)

    logging.debug('number of missing values for each column containing them is: {}'.format(len(s[s > 0])))

    # Drop the rows that have missing values.
    df_profeat.dropna(inplace=True)
    logging.debug('number of rows remaining, without NaNs: {:,}'.format(len(df_profeat)))
    
    return {'df_dragon_features': df_dragon_features,
           'df_fingerprints': df_fingerprints,
           'df_binding_sites': df_binding_sites,
           'df_binding_sites_corona': df_binding_sites_corona,
           'df_profeat': df_profeat,
           'df_profeat_corona': df_profeat_corona}


  def __create_features(self, feature_sets, df_interactions):
    # Get the individual feature sets
    df_dragon_features = feature_sets['df_dragon_features']
    df_fingerprints = feature_sets['df_fingerprints']
    df_binding_sites = feature_sets['df_binding_sites']
    df_profeat = feature_sets['df_profeat']
    
    
    # Form the complete feature set by joining the data frames according to _cid_ and _pid_.
    # See the data readme in the Gitbug repository:
    # https://github.com/BrianDavisMath/FDA-COVID19/tree/master/data.
    
    # By convention, the file features should be concatenated in the following order (for consistency):
    # **binding_sites**, **profeat**, **dragon_features**, **fingerprints**.
    
    df_features = pd.merge(df_interactions, df_binding_sites, on='pid', how='inner')
    
    
    df_features = pd.merge(df_features, df_profeat, on='pid', how='inner')
    
    df_dragon_features.index.name = 'cid'
    df_features = pd.merge(df_features, df_dragon_features, on='cid', how='inner')
    
    df_features = pd.merge(df_features, df_fingerprints, on='cid', how='inner')
    
    # release memory used by previous dataframes.
    del df_binding_sites
    del df_profeat
    del df_dragon_features
    del df_fingerprints
    
    return df_features


  '''
  ==================================================================
  Functions for using XGBoost classification
  ==================================================================
  '''
  def __train_and_eval(self, df_in, df_validation, xgb, sample_weight=None):
    Y = df_in['activity'].values
    
    X = df_in.loc[:, ~df_in.columns.isin(self.non_feature_columns)]
    X.columns = list(range(0, len(X.columns)))

    model = self.__train(X, Y, xgb=xgb, sample_weight=sample_weight)
    del X
    del Y

    # load test data and gather metrics
    results = self.__gather_metrics(df_in, model, df_validation)

    del model

    return results


  # train an XGBoost model and use an internal split for evaluation.
  def __train(self, X, Y, xgb=None, sample_weight=None):
    xgb.fit(X, Y, verbose=False, sample_weight=sample_weight)
    return xgb


  # Return p@k where k = top 10% of results ordered by product of weight and probability of activity.
  def __target_metric(self, probabilities, df_validation):
    prediction_model_probs = probabilities[:, 1]
    df_validation = df_validation.copy()
    weights = df_validation['weight']

    # Multiply the probability by the weight and sort to get p@k based on this.
    df_validation['prob_weight'] = prediction_model_probs*weights
    df_validation.sort_values(by='prob_weight', ascending=False, inplace=True)

    # Get the top 10% of results.
    k = int(len(df_validation)*0.1)

    results = []

    v = df_validation[:k][(df_validation['activity'] == 1.0)]
    p_at_k = v['activity'].sum()/k
    
    del df_validation

    return p_at_k


  # run model against the validation/test set and return the 
  # target metric along with accuracy, precision and recall.
  def __gather_metrics(self, df_in, model, df_validation):
    # Take only the reduced set of columns.
    df_val = df_validation[df_in.columns.tolist()].copy()

    y_test = df_val['activity'].values
    X_test = df_val.loc[:, ~df_val.columns.isin(self.non_feature_columns)]

    # accuracy, precision and recall
    X_test.columns = list(range(0, len(X_test.columns)))

    # make predictions for test data
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    accuracy = sum(np.diag(cm))/cm.sum()*100
    precision = cm[1][1]/sum(cm[:, 1])*100
    recall = cm[1][1]/sum(cm[1, :])*100

    probabilities = model.predict_proba(X_test)
    target_metric = self.__target_metric(probabilities, df_validation)

    del df_val
    del X_test

    return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'target_metric': target_metric}


def main(argv):
  training_features = None
  validation_features = None
  data_folder = 'data/'
  num_runs = 10
  k = 1
  job_name = ''
  generate_cont_features = False

  
  # check arguments.
  try:
    opts, args = getopt.getopt(argv,"hf:n:r:c:",["data=", "name=", "runs=", "cfeat="])
  except getopt.GetoptError:
    print('\n\n')
    logging.debug('job.py -f <data_folder> -n <job_name> -r <number of runs> -c <generate continuous features>')
    print('\n\n')
    sys.exit(2)

  if len(opts) < 2:
    print('\n\n')
    logging.debug('job.py -f <data_folder> -n <job_name> -r <number of runs> -c <generate continuous features>')
    print('\n\n')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('\n\n')
      logging.debug('job.py -f <data_folder> -n <job_name> -r <number of runs> -c <generate continuous features>')
      print('\n\n')
      sys.exit()
    elif opt in ("-f", "--data"):
      data_folder = arg
    elif opt in ("-n", "--name"):
      job_name = arg
    elif opt in ("-r", "--runs"):
      num_runs = int(arg)
    elif opt in ("-c", "--cfeat"):
      generate_cont_features = arg is not None

  logging.debug('\ndata in {}'.format(data_folder))
  logging.debug('job name {}'.format(job_name))
  logging.debug('runs {}'.format(num_runs))
  logging.debug('generate_cont_features {}\n'.format(generate_cont_features))

  xgb = XGBoostClassifier(
    data_folder=data_folder, 
    job_name=job_name,
    num_runs=num_runs,
    generate_cont_features=generate_cont_features)


if __name__== "__main__":
  main(sys.argv[1:])



