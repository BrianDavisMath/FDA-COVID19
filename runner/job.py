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
  
  [-a] max_activity_threshold - used to sub-sample the training data for the
  first run

  [-s] activity_threshold_step - the amount to reduced the activity threshold
  on each run before sub-sampling the training data

  [-f] data_folder - default /data. Specifies the location of the data from which the
  features will be selected.

  [-d] use_dimension_reduction_weights - [True|False] whether to use sample_activity_score 
  for sample_weight when training XGBoost for feature selection

  [-t] use_training_weights - [True|False] whether to use sample_activity_score 
  for sample_weight when training XGBoost for activity classification


Outputs:

  results.csv - model outputs including binary classification for activity 
  as well as probability of activity for each cid/pid pair in the
  validation set;

  metrics.csv - the accuracy, precision, recall, weighted and unweighted
  F1 score representing model performane

  important_features.csv - the set of most important features, along with
  their information gain values, as returned by the first run of XGBoost.
  These are the features that are selected for the classification pass.


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
    | | |____expasy.csv
    | |____training_validation_split
    | | |____training_interactions.csv
    | | |____validation_interactions.csv


Example call to run program:

  job.py -a 0.05 -s 0.01 -f test_data/ -d False -t False

'''
import sys, getopt
import os
import pandas as pd
import numpy as np
import h5py

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

class XGBoostClassifier():
    
  def __init__(
    self, 
    max_activity_threshold, 
    activity_threshold_step,
    data_folder,
    use_dimension_reduction_weights,
    use_training_weights):
      self.max_activity_threshold = float(max_activity_threshold)
      self.activity_threshold_step = float(activity_threshold_step)
      self.data_loc = data_folder

      self.bad_dragon_cols = []

      # XGBoost parameters
      self.learning_rate=0.02
      self.n_estimators=600
      self.objective='binary:logistic'
      self.subsample=0.6
      self.min_child_weight=1
      self.max_depth=6
      self.gamma=5
      self.colsample_bytree=0.8

      # Create the feature sets for training and validation
      # Get the individual feature sets
      self.feature_sets = self.__load_feature_files()
      self.training_features = self.__create_features('training_interactions.csv', 
        'training_features.h5', self.feature_sets)
      self.validation_features = self.__create_features('validation_interactions.csv', 
        'validation_features.h5', self.feature_sets)

      # get the validation features
      store = pd.HDFStore(self.data_loc + 'validation_features.h5')
      df_validation = pd.DataFrame(store['df' ])
      store.close()

      print('\n\n Validation features:')
      print(df_validation.head())
      print('\n\ndf_validation - rows: {:,}, columns: {:,}\n\n'.format(len(df_validation), len(df_validation.columns)))

      # Iterate over activity thresholds and produce results for each one
      activity_threshold = self.max_activity_threshold
      run_id = 0
      while activity_threshold > 0.0:
        df_features = self.training_features[self.training_features['sample_activity_score'] > activity_threshold]
        print('\n\nsample_activity_score ({}) features shape: {}'
          .format(activity_threshold, df_features.shape))

        # drop zero-variance columns
        var_cols = [col for col in df_features.columns if df_features[col].nunique() > 1]
        df = df_features[var_cols].copy()

        print('Dropped {:,} columns that have zero variance.'.format(len(df_features.columns)-len(var_cols)))

        del df_features
        df_features = df

        print('Shape after dropping zero-variance columns - rows: {:,}, columns: {:,}'.
          format(len(df_features), len(df_features.columns)))


        # Get important features using XGBoost
        Y = df_features['activity'].values
        X = df_features.copy()
        self.__drop_non_features(X)
        print('X with non-features dropped - rows: {:,}, columns: {:,}'.format(len(X), len(X.columns)))
        X.columns = list(range(0, len(X.columns)))

        # train
        sample_weight = None
        if use_dimension_reduction_weights == True:
          sample_weight = df_features['sample_activity_score']
        xgb = self.__train(X, Y, xgb=self.__get_xgb(), sample_weight=sample_weight)

        # get features
        xgb_features = self.__get_features(xgb, df_features)
        top_feature_cols = list(xgb_features['feature'].values)
        print('number of most important features: {:,}'.format(len(xgb_features)))
        df = df_features[['cid', 'pid', 'activity', 'sample_activity_score']+top_feature_cols]
        del df_features
        df_features = df

        # cid set with subset of features derived from previous cid/pid combined dimension reduction
        drug_column_names = self.__get_drug_column_names()
        cid_features = [col for col in top_feature_cols if col in drug_column_names]

        df_drugs = df_features[['cid', 'pid', 'activity', 'sample_activity_score']+cid_features]
        print('df_drugs - rows: {:,}, columns: {:,}'.format(len(df_drugs), len(df_drugs.columns)))
        print('\ncid features:\n')
        print(df_drugs.head())

        # pid set with subset of features derived from previous cid/pid combined dimension reduction
        protein_column_names = self.__get_protein_column_names()
        pid_features = [col for col in top_feature_cols if col in protein_column_names]
        df_proteins = df_features[['cid', 'pid', 'activity', 'sample_activity_score']+pid_features]
        print('df_proteins - rows: {:,}, columns: {:,}'.format(len(df_proteins), len(df_proteins.columns)))
        print('\npid features:\n')
        print(df_proteins.head())

        # Run models
        print('cid/pid combined with activity score weighting, results:\n')
        combined_model_results = self.__train_and_eval(df_features, df_validation, use_weights=use_training_weights)

        print('\ncid with activity score weighting, results:\n')
        drugs_model_results = self.__train_and_eval(df_drugs, df_validation, use_weights=use_training_weights)

        print('\npid combined with activity score weighting, results:\n')
        proteins_model_results = self.__train_and_eval(df_proteins, df_validation, use_weights=use_training_weights)

        del df_features
        del df_drugs
        del df_proteins

        self.__results_to_csv(
          activity_threshold,
          run_id,
          use_dimension_reduction_weights,
          use_training_weights,
          combined_model_results,
          drugs_model_results,
          proteins_model_results)

        run_id = run_id+1
        activity_threshold = round(activity_threshold - self.activity_threshold_step, 4)


  '''
  ==================================================================
  Functions for stitching together the individual features CSV files
  to create the training and validation sets.
  ==================================================================
  '''

  # Write modeling results to CSV
  def __results_to_csv(
    self, 
    threshold,
    run_id,
    used_dimension_reduction_weights,
    used_training_weights,
    combined_model_results,
    drugs_model_results,
    proteins_model_results):

    df_validation = combined_model_results['df_validation']
    df_validation.reset_index(inplace=True, drop=True)
    validation_weights = combined_model_results['validation_weights']

    print('validation observations:\n\n')
    print(df_validation.head())

    results = []

    for index, row in df_validation.iterrows():
      result = []
      result.append(run_id)
      result.append(threshold)
      result.append(used_dimension_reduction_weights)
      result.append(used_training_weights)
      result.append(row['cid'])
      result.append(row['pid'])
      result.append(row['activity'])
      result.append(row['sample_activity_score'])
      result.append(validation_weights[index])

      cid_only_probability = drugs_model_results['probabilities'][index][1]
      pid_only_probability = proteins_model_results['probabilities'][index][1]
      combined_probability = combined_model_results['probabilities'][index][1]

      result.append(cid_only_probability)
      result.append(pid_only_probability)
      result.append(combined_probability)

      result.append(self.learning_rate)
      result.append(self.n_estimators)
      result.append(self.objective)
      result.append(self.subsample)
      result.append(self.min_child_weight)
      result.append(self.max_depth)
      result.append(self.gamma)
      result.append(self.colsample_bytree)

      results.append(result)

    df = pd.DataFrame(results, columns=[
      'run_id',
      'run_threshold',
      'used_dim_red_weights',
      'used_training_weights',
      'cid',
      'pid',
      'activity',
      'sample_activity_score',
      'validation_weight',
      'cid_only_predict_proba',
      'pid_only_predict_proba',
      'combined_predict_proba',
      'xgb_learning_rate',
      'xgb_n_estimators',
      'xgb_objective',
      'xgb_subsample',
      'xgb_min_child_weight',
      'xgb_max_depth',
      'xgb_gamma',
      'xgb_colsample_bytree'
      ])

    path = 'results'
    try:
      os.mkdir(path)
    except OSError:
      print ("Creation of the directory %s failed" % path)
    else:
      print ("Successfully created the directory %s " % path)

    df.to_csv('results/results_{}.csv'.format(run_id), index=False)
    del df


  # load a specific features CSV file
  def __load_data(self, path, data_type=None):
    if data_type:
        df = pd.read_csv(path, index_col=0, dtype=data_type)
    else:
        df = pd.read_csv(path, index_col=0)
    print('Number of rows: {:,}\n'.format(len(df)))
    print('Number of columns: {:,}\n'.format(len(df.columns)))
    
    columns_missing_values = df.columns[df.isnull().any()].tolist()
    print('{} columns with missing values\n\n'.format(len(columns_missing_values)))
    
    cols = df.columns.tolist()
    column_types = [{col: df.dtypes[col].name} for col in cols][:10]
    print('column types:\n')
    print(column_types, '\n\n')
    print(df.head(2))
    
    return df

  # print out summary after each features merge
  def __print_merge_details(self, df_merge_result, df1_name, df2_name):
    print('Joining {} on protein {} yields {:,} rows and {:,} columns'. \
          format(df1_name, df2_name, len(df_merge_result), 
          len(df_merge_result.columns)))

  # Get the individual feature sets as data frames
  def __load_feature_files(self):
    print('===============================================')
    print('\ndragon_features.csv')
    print('===============================================')
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

    print('number of columns where the frequency of "na" values is <= {}%: {}.'.format(pct_threshold, len(ok_cols)))
    
    df_dragon_features = df_dragon_features[ok_cols].copy()

    # convert all values except "na"s to numbers and set "na" values to NaNs.
    df_dragon_features = df_dragon_features.apply(pd.to_numeric, errors='coerce')

    columns_missing_values = df_dragon_features.columns[df_dragon_features.isnull().any()].tolist()
    print('{} columns with missing values.\n\n'.format(len(columns_missing_values)))

    # replace NaNs with column means
    df_dragon_features.fillna(df_dragon_features.mean(), inplace=True)

    columns_missing_values = df_dragon_features.columns[df_dragon_features.isnull().any()].tolist()
    print('{} columns with missing values (after imputing): {}\n\n'.format(len(columns_missing_values), 
                                                                       columns_missing_values))
    print('===============================================')
    print('fingerprints.csv')
    print('===============================================')
    df_fingerprints = self.__load_data(self.data_loc+'drug_features/fingerprints.csv')
    
    print('===============================================')
    print('binding_sites.csv')
    print('===============================================')
    df_binding_sites = self.__load_data(self.data_loc+'protein_features/binding_sites.csv')
    
    # Name the index to 'pid' to allow joining to other feaure files later.
    df_binding_sites.index.name = 'pid'
    
    print('===============================================')
    print('expasy.csv')
    print('===============================================')
    df_expasy = self.__load_data(self.data_loc+'protein_features/expasy.csv')
    
    print('===============================================')
    print('profeat.csv')
    print('===============================================')
    df_profeat = self.__load_data(self.data_loc+'protein_features/profeat.csv')
    
    # Name the index to 'pid' to allow joining to other feaure files later.
    df_profeat.index.name = 'pid'
    
    # profeat has some missing values.
    s = df_profeat.isnull().sum(axis = 0)

    print('number of missing values for each column containing them is: {}'.format(len(s[s > 0])))

    # Drop the rows that have missing values.
    df_profeat.dropna(inplace=True)
    print('number of rows remaining, without NaNs: {:,}'.format(len(df_profeat)))
    
    return {'df_dragon_features': df_dragon_features,
           'df_fingerprints': df_fingerprints,
           'df_binding_sites': df_binding_sites,
           'df_expasy': df_expasy,
           'df_profeat': df_profeat}


  def __create_features(self, split_file_name, out_file_name, feature_sets):
    print('===============================================')
    print('interactions.csv')
    print('===============================================')
    
    # load interactions.csv
    df_interactions = self.__load_data(
      self.data_loc+'training_validation_split/' + split_file_name)

    # Rename the 'canonical_cid' column simply to 'cid' to simplifiy joining to the other feature sets later.
    df_interactions.rename(columns={"canonical_cid": "cid"}, inplace=True)
    print(df_interactions.head())
    
    # Get the individual feature sets
    df_dragon_features = feature_sets['df_dragon_features']
    df_fingerprints = feature_sets['df_fingerprints']
    df_binding_sites = feature_sets['df_binding_sites']
    df_expasy = feature_sets['df_expasy']
    df_profeat = feature_sets['df_profeat']
    
    print('\n\n===============================================')
    print('Join the data using {}\n'.format(split_file_name))
    print('===============================================')
    
    # Form the complete feature set by joining the data frames according to _cid_ and _pid_.
    # See the data readme in the Gitbug repository:
    # https://github.com/BrianDavisMath/FDA-COVID19/tree/master/data.
    
    # By convention, the file features should be concatenated in the following order (for consistency):
    # **binding_sites**, **expasy**, **profeat**, **dragon_features**, **fingerprints**.
    
    print('\n\n-----------------------------------------------')
    print('df_interactions + df_binding_sites = df_features \n')
    df_features = pd.merge(df_interactions, df_binding_sites, on='pid', how='inner')
    self.__print_merge_details(df_features, 'interactions', 'binding_sites')
    
    print('\n\n-----------------------------------------------')
    print('df_features + df_expasy \n')
    df_features = pd.merge(df_features, df_expasy, on='pid', how='inner')
    self.__print_merge_details(df_features, 'features', 'expasy')
    
    print('\n\n-----------------------------------------------')
    print('df_features + df_profeat \n')
    df_features = pd.merge(df_features, df_profeat, on='pid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_profeat')
    
    print('\n\n-----------------------------------------------')
    print('df_features + df_dragon_features \n')
    df_dragon_features.index.name = 'cid'
    df_features = pd.merge(df_features, df_dragon_features, on='cid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_dragon_features')
    
    print('\n\n-----------------------------------------------')
    print('df_features + df_fingerprints \n')
    df_features = pd.merge(df_features, df_fingerprints, on='cid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_fingerprints')
    
    print('\n\n-----------------------------------------------')
    print('Number of rows in joined feature set: {:,}\n'.format(len(df_features)))
    print('Number of columns in joined feature set: {:,}\n'.format(len(df_features.columns)))
    
    # release memory used by previous dataframes.
    del df_interactions
    del df_binding_sites
    del df_expasy
    del df_profeat
    del df_dragon_features
    del df_fingerprints
    
    # Save features to file
    store = pd.HDFStore(self.data_loc + out_file_name)
    store['df'] = df_features
    store.close()

    return df_features


  '''
  ==================================================================
  Functions for using XGBoost for dimension reduction and classification
  ==================================================================
  '''

  # train an XGBoost model and use an internal split for evaluation.
  def __train(self, X, Y, xgb=None, sample_weight=None):
    xgb.fit(X, Y, verbose=False, sample_weight=sample_weight)
    return xgb

  # Get feature importance as information gain 
  # (the improvement in accuracy brought by a feature) from an XGBoost model.
  def __get_features(self, model, df_features):
    gain_importance = model.get_booster().get_score(importance_type="gain")
    feature_indices = [int(key) for key in gain_importance.keys()]

    df = df_features.copy()
    self.__drop_non_features(df)
    top_feature_cols = df.columns.values[feature_indices] # turn numbers back into column names
    del df

    return pd.DataFrame({'feature': top_feature_cols, 
      'importance': list(gain_importance.values())}, columns = ['feature', 'importance'])

  # calculate the sample weights used for the F1 Score
  # TODO: Brian to write documentation for this:
  def __get_validation_weights(self, training_features_active, training_features_inactive,
                             validation_features_active, validation_features_inactive):
    active_nbrs = NearestNeighbors(n_neighbors=1).fit(training_features_active)
    inactive_nbrs = NearestNeighbors(n_neighbors=1).fit(training_features_inactive)
    act_act_distances, _ = active_nbrs.kneighbors(validation_features_active)
    inact_act_distances, _ = active_nbrs.kneighbors(validation_features_inactive)
    act_inact_distances, _ = inactive_nbrs.kneighbors(validation_features_active)
    inact_inact_distances, _ = inactive_nbrs.kneighbors(validation_features_inactive)
    active_scores = act_act_distances / act_inact_distances
    active_weights = (1 + np.argsort(np.argsort(active_scores.flatten()))) / len(active_scores)
    inactive_scores = inact_inact_distances / inact_act_distances
    inactive_weights = (1 + np.argsort(np.argsort(inactive_scores.flatten()))) / len(inactive_scores)
    return active_weights, inactive_weights

  # get only the feature names from a csv file
  def __get_csv_feature_names(self, file):
    columns = pd.read_csv(self.data_loc + file, index_col=0, nrows=1).columns.tolist()
    print('{}: columns: {:,}'.format(file, len(columns)))
    return columns

  def __get_h5_feature_names(self, file):
    store = pd.HDFStore(self.data_loc + file)
    meta = store.select('df', start=1, stop=1)
    store.close()
    print('{}: columns: {:,}'.format(file, len(meta.columns)))
    return list(meta.columns)

  def __drop_non_features(self, df):
    if 'index' in df:
        df.drop('index', axis=1, inplace=True)
    if 'activity' in df:
        df.drop('activity', axis=1, inplace=True)
    if 'cid' in df:
        df.drop('cid', axis=1, inplace=True)
    if 'pid' in df:
        df.drop('pid', axis=1, inplace=True)
    if 'expanding_mean' in df:
        df.drop('expanding_mean', axis=1, inplace=True)
    if 'sample_activity_score' in df:
        df.drop('sample_activity_score', axis=1, inplace=True)

  # load test data and gather metrics
  def __gather_metrics(self, df_in, model, df_validation):
    # Take only the reduced set of columns.
    df_validation = df_validation[df_in.columns.tolist()]

    print('rows: {:,}, columns: {:,}'.format(len(df_validation), len(df_validation.columns)))

    # weighted F1 score
    training_features_active = df_in[df_in['activity']==1]
    training_features_inactive = df_in[df_in['activity']==0]
    
    self.__drop_non_features(training_features_active)
    self.__drop_non_features(training_features_inactive)

    validation_features_active = df_validation[df_validation['activity']==1]
    validation_features_inactive = df_validation[df_validation['activity']==0]

    # Needed later for reporting
    df_validation_features = validation_features_active.append(validation_features_inactive).copy()

    y_test = df_validation_features['activity'].values

    self.__drop_non_features(validation_features_active)
    self.__drop_non_features(validation_features_inactive)

    X_test = validation_features_active.append(validation_features_inactive)

    # accuracy, precision and recall
    X_test.columns = list(range(0, len(X_test.columns)))

    # make predictions for test data
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    accuracy = sum(np.diag(cm))/cm.sum()*100
    precision = cm[1][1]/sum(cm[:, 1])*100
    recall = cm[1][1]/sum(cm[1, :])*100

    print("Accuracy = {:0.2f}%".format(accuracy))
    print("Precision = {:0.2f}%".format(precision))
    print("Recall = {:0.2f}%".format(recall))

    active_weights, inactive_weights = self.__get_validation_weights(
        training_features_active, 
        training_features_inactive,
        validation_features_active, 
        validation_features_inactive)

    assert(len(inactive_weights) + len(active_weights) == len(df_validation))

    weights = list(active_weights) + list(inactive_weights)

    f1score = f1_score(y_test, y_pred, average='binary', sample_weight=weights)

    print('F1 Score (weighted): {:0.2f}%'.format(f1score*100))
    
    f1score_unweighted = f1_score(y_test, y_pred, average='binary')
    print('F1 Score (unweighted): {:0.2f}%'.format(f1score_unweighted*100))
    
    probabilities = model.predict_proba(X_test)

    return {
      'probabilities': probabilities, 
      'validation_weights': weights,
      'df_validation': df_validation_features,
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1_score_weighted': f1score,
      'f1_score_unweighted': f1score_unweighted}
      
      
  # load drug data column names
  def __get_drug_column_names(self):
    cid_dragon_features = self.__get_csv_feature_names('drug_features/dragon_features.csv')

    # We renamed the dragon features columns in the greg-features notebook.
    cid_dragon_features = ['cid_'+col for col in cid_dragon_features]

    # We dropped columns that are < 2% filled with values in the greg-features notebook.
    empty_cols = self.bad_dragon_cols
    cid_dragon_features = [col for col in cid_dragon_features if col not in empty_cols]

    cid_fingerprints = self.__get_csv_feature_names('drug_features/fingerprints.csv')

    cid_features = cid_dragon_features + cid_fingerprints
    print()
    print('{:,} drug featues'.format(len(cid_features)))
    return cid_features


  # load protein data column names
  def __get_protein_column_names(self):
    pid_binding_sites = self.__get_csv_feature_names('protein_features/binding_sites.csv')
    pid_expasy = self.__get_csv_feature_names('protein_features/expasy.csv')
    pid_profeat = self.__get_csv_feature_names('protein_features/profeat.csv')

    pid_features = pid_binding_sites + pid_expasy + pid_profeat
    print()
    print('{:,} protein featues'.format(len(pid_features)))
    return pid_features

  def __get_xgb(self):
    xgb = XGBClassifier(learning_rate=self.learning_rate, 
                        n_estimators=self.n_estimators, 
                        objective=self.objective,
                        subsample=self.subsample,
                        min_child_weight=self.min_child_weight,
                        max_depth=self.max_depth,
                        gamma=self.gamma,
                        colsample_bytree=self.colsample_bytree)
    return xgb

  # Model for combined cid/pid features using activity scores as sample weights.
  def __train_and_eval(self, df_in, df_validation, use_weights=True):
    Y = df_in['activity'].values
    df = df_in.copy()
    self.__drop_non_features(df)
    X = df
    X.columns = list(range(0, len(X.columns)))

    xgb = self.__get_xgb()

    sample_weight = None
    if use_weights == True:
      sample_weight = df_in['sample_activity_score']

    model = self.__train(X, Y, xgb=xgb, sample_weight=sample_weight)
    del df

    # load test data and gather metrics
    results = self.__gather_metrics(df_in, model, df_validation)
    return results


def main(argv):
  training_features = None
  validation_features = None
  max_activity_threshold = None
  activity_threshold_step = None
  data_folder = 'data/'
  use_dimension_reduction_weights = 'True'
  use_training_weights = 'True'

  
  # check arguments.
  try:
    opts, args = getopt.getopt(argv,"ha:s:f:d:t:",["athresh=", "step=", "data=", "dweights=", "tweights="])
  except getopt.GetoptError:
    print('\n\njob.py -a <max_activity_threshold> -s <activity_threshold_step> \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights>\n\n')
    sys.exit(2)

  if len(opts) < 2:
    print('\n\njob.py -a <max_activity_threshold> -s <activity_threshold_step> \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights>\n\n')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('\n\njob.py -a <max_activity_threshold> -s <activity_threshold_step> \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights>\n\n')
      sys.exit()
    elif opt in ("-a", "--athresh"):
      max_activity_threshold = arg
    elif opt in ("-s", "--step"):
      activity_threshold_step = arg
    elif opt in ("-f", "--data"):
      data_folder = arg
    elif opt in ("-d", "--dweights"):
      use_dimension_reduction_weights = arg
    elif opt in ("-t", "--tweights"):
      use_training_weights = arg



  if max_activity_threshold is None or activity_threshold_step is None:
    print('job.py -a <max_activity_threshold> -s <activity_threshold_step> \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights>\n\n')
    sys.exit()

  print('\n\nmax_activity_threshold is {}'.format(max_activity_threshold))
  print('activity_threshold_step is {}'.format(activity_threshold_step))
  print('use_dimension_reduction_weights is {}'.format(use_dimension_reduction_weights=='True'))
  print('use_training_weights is {}\n\n'.format(use_training_weights=='True'))

  xgb = XGBoostClassifier(
    max_activity_threshold=max_activity_threshold,
    activity_threshold_step=activity_threshold_step,
    data_folder=data_folder, 
    use_dimension_reduction_weights=use_dimension_reduction_weights=='True',
    use_training_weights=use_training_weights=='True')


if __name__== "__main__":
  main(sys.argv[1:])




