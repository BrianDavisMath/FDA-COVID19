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

  [-m] activity_threshold_stop - the value (+ activity_threshold_step) upon which the run
  terminates

  [-f] data_folder - defaults to "/data". Specifies the location of the data from which the
  features will be selected.

  [-d] use_dimension_reduction_weights - [True|False] whether to use activity_score 
  for sample_weight when training XGBoost for feature selection

  [-t] use_training_weights - [True|False] whether to use activity_score 
  for sample_weight when training XGBoost for activity classification

  [-n] name - job name used to place results in a directory of that name


Outputs:

  results_[run_id].csv - model outputs including binary classification for activity 
  as well as probability of activity for each cid/pid pair in the
  validation set;

  params_and_metrics.csv - the accuracy, precision, recall, weighted and unweighted
  F1 score representing model performane along with the XGBoost parameters used

  feature_importances_[run_id].csv - the set of most important features, along with
  their information gain values, as returned by the first run of XGBoost for dimension
  reduction as well as for each classification (cid-only, pid-only and combined).

  Note: all results files are written to a dynamically created **results folder**.
  Also, **run_id** is a zero-based index of each run and corresponds to each unique
  activity_threshold used in sampling the training data


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

  python job.py -a 0.02 -s 0.01 -f test_data/ -d False -t False -n 'test_0'

  This will run two experiments where the activity threshold, used for sampling
  the training data, are 0.02 and 0.01. The corresponding run_ids will be 0 and 1.

'''
import logging
import sys, getopt
import os
import gc
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(filename='job.log',level=logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logging.getLogger().addHandler(ch)


class WeightedROCAUC:
  def __init__(self, pid_only_predict_probs, cid_only_predict_probs, validation_labels, classifier):
    self.pid_probs = pid_only_predict_probs
    self.cid_probs = cid_only_predict_probs
    self.labels = validation_labels
    self.classifier = classifier
    pid_cid_probs = np.vstack([self.pid_probs, self.cid_probs]).transpose()
    composite_model = LogisticRegression(random_state=0).fit(pid_cid_probs, self.labels)
    self.composite_predictions = composite_model.predict_proba(pid_cid_probs)[:, 1]

    self.composite_gain_ = np.vectorize(self.composite_gain_)
  
  def composite_gain_(self, prediction_loss, composite_loss):
    if (prediction_loss == 0) and (composite_loss == 0):
        return 0
    else:
        return 0.5 * (1 + (composite_loss - prediction_loss) / (composite_loss + prediction_loss))

  def score(self, prediction_model_probs, max_loss=5.0):
    prediction_loss = self.classifier.log_loss_(self.labels, prediction_model_probs)
    prediction_loss = np.clip(prediction_loss, 0.0, max_loss)
    composite_loss = self.classifier.log_loss_(self.labels, self.composite_predictions)
    composite_loss = np.clip(composite_loss, 0.0, max_loss)

    validation_weights = self.composite_gain_(prediction_loss, composite_loss)
    return roc_auc_score(self.labels, prediction_model_probs, sample_weight=validation_weights)


class XGBoostClassifier():
    
  def __init__(
    self, 
    max_activity_threshold, 
    activity_threshold_step,
    activity_threshold_stop,
    data_folder,
    use_dimension_reduction_weights,
    use_training_weights,
    job_name):

      self.max_activity_threshold = float(max_activity_threshold)
      self.activity_threshold_step = float(activity_threshold_step)
      self.activity_threshold_stop = float(activity_threshold_stop)
      self.data_loc = data_folder
      self.job_name = job_name

      self.log_loss_ = np.vectorize(self.log_loss_)

      self.bad_dragon_cols = []
      self.non_feature_columns = ['activity', 'cid', 'pid', 'activity_score']

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
      df_validation = self.validation_features

      logging.debug(' Validation features:')
      logging.debug(df_validation.head())
      logging.debug('df_validation - rows: {:,}, columns: {:,}'.format(len(df_validation), len(df_validation.columns)))

      # Iterate over activity thresholds and produce results for each one
      activity_threshold = self.max_activity_threshold
      run_id = 0
      combined_results = []
      drugs_results = []
      proteins_results = []
      thresholds = []
      roc_results = []

      gc.collect()

      while activity_threshold > self.activity_threshold_stop:
        df_features = self.training_features[self.training_features['activity_score'] > activity_threshold]
        logging.debug('activity_score ({}) features shape: {}'
          .format(activity_threshold, df_features.shape))

        # drop zero-variance columns
        var_cols = [col for col in df_features.columns if df_features[col].nunique() > 1]
        df = df_features[var_cols]

        logging.debug('Dropped {:,} columns that have zero variance.'.format(len(df_features.columns)-len(var_cols)))

        del df_features
        df_features = df

        logging.debug('Shape after dropping zero-variance columns - rows: {:,}, columns: {:,}'.
          format(len(df_features), len(df_features.columns)))


        # Get important features using XGBoost
        Y = df_features['activity'].values
        X = df_features.loc[:, ~df_features.columns.isin(self.non_feature_columns)]

        print('\n\n X: \n\n')
        print(X.head())

        logging.debug('X with non-features dropped - rows: {:,}, columns: {:,}'.format(len(X), len(X.columns)))
        X.columns = list(range(0, len(X.columns)))

        self.set_pos_weight_param(df_features)

        # train
        sample_weight = None
        if use_dimension_reduction_weights == True:
          sample_weight = df_features['activity_score']
        xgb = self.__train(X, Y, xgb=self.__get_xgb(), sample_weight=sample_weight)

        # get features
        xgb_features = self.__get_features(xgb, df_features)
        top_feature_cols = list(xgb_features['feature'].values)
        logging.debug('number of most important features: {:,}'.format(len(xgb_features)))
        df = df_features[['cid', 'pid', 'activity', 'activity_score']+top_feature_cols]
        del df_features
        df_features = df

        # cid set with subset of features derived from previous cid/pid combined dimension reduction
        drug_column_names = self.__get_drug_column_names()
        cid_features = [col for col in top_feature_cols if col in drug_column_names]

        df_drugs = df_features[['cid', 'pid', 'activity', 'activity_score']+cid_features]
        logging.debug('df_drugs - rows: {:,}, columns: {:,}'.format(len(df_drugs), len(df_drugs.columns)))
        logging.debug('cid features:')
        logging.debug(df_drugs.head())

        # pid set with subset of features derived from previous cid/pid combined dimension reduction
        protein_column_names = self.__get_protein_column_names()
        pid_features = [col for col in top_feature_cols if col in protein_column_names]
        df_proteins = df_features[['cid', 'pid', 'activity', 'activity_score']+pid_features]
        logging.debug('df_proteins - rows: {:,}, columns: {:,}'.format(len(df_proteins), len(df_proteins.columns)))
        logging.debug('pid features:')
        logging.debug(df_proteins.head())

        # Run models
        logging.debug('cid with activity score weighting, results:')
        drugs_model_results = self.__train_and_eval(df_drugs, df_validation, use_weights=False)

        logging.debug('pid combined with activity score weighting, results:')
        proteins_model_results = self.__train_and_eval(df_proteins, df_validation, use_weights=False)

        # Training weights
        pid_only_predict_train_probs = proteins_model_results['training_probabilities'][:, 1]
        cid_only_predict_train_probs = drugs_model_results['training_probabilities'][:, 1]
        training_labels = df_features['activity'].values

        pid_only_predict_probs = proteins_model_results['probabilities'][:, 1]
        cid_only_predict_probs = drugs_model_results['probabilities'][:, 1]
        validation_labels = df_validation['activity'].values

        self.training_weights = self.generate_training_weights(
          pid_only_predict_train_probs, 
          cid_only_predict_train_probs,
          training_labels, max_loss=5.0)

        # Combined model results
        logging.debug('cid/pid combined with activity score weighting, results:')
        combined_model_results = self.__train_and_eval(df_features, df_validation, use_weights=use_training_weights)

        # Target metric
        prediction_model_probs = combined_model_results['probabilities'][:, 1]
        self.weightedROCAUC = WeightedROCAUC(pid_only_predict_probs, 
          cid_only_predict_probs, validation_labels, self)

        roc_result = self.weightedROCAUC.score(prediction_model_probs, max_loss=5.0)
        roc_results.append(roc_result)

        combined_results.append(combined_model_results)
        drugs_results.append(drugs_model_results)
        proteins_results.append(proteins_model_results)

        self.__results_to_csv(
          activity_threshold,
          run_id,
          use_dimension_reduction_weights,
          use_training_weights,
          combined_model_results,
          drugs_model_results,
          proteins_model_results)

        self.__importance_to_csv(
          run_id,
          xgb_features,
          combined_model_results['model'],
          drugs_model_results['model'],
          proteins_model_results['model'],
          df_features,
          df_drugs,
          df_proteins)

        del df_features
        del df_drugs
        del df_proteins
        del combined_model_results['model']
        del drugs_model_results['model']
        del proteins_model_results['model']

        run_id = run_id+1
        thresholds.append(activity_threshold)
        activity_threshold = round(activity_threshold - self.activity_threshold_step, 4)
        gc.collect()

      self.__params_and_metrics_to_csv(
        thresholds,
        use_dimension_reduction_weights,
        use_training_weights,
        combined_results,
        drugs_results,
        proteins_results,
        roc_results)


  def log_loss_(self, true_label, predicted):
    if true_label == 1:
        return -np.log(predicted)
    else:
        return -np.log(1 - predicted)

  def generate_training_weights(self, pid_only_predict_probs, cid_only_predict_prob, training_labels, max_loss=5.0):
    # For use on training data
    X_pid_cid = np.vstack([pid_only_predict_probs, cid_only_predict_prob]).transpose()
    pid_cid_model = LogisticRegression(random_state=0).fit(X_pid_cid, training_labels)
    pid_cid_predictions = pid_cid_model.predict_proba(X_pid_cid)[:, 1]
    pid_cid_loss = self.log_loss_(training_labels, pid_cid_predictions)
    return np.clip(pid_cid_loss, 0.0, max_loss)

  def set_pos_weight_param(self, df):
    num_active = len(df[df['activity']==1.0])
    num_inactive = len(df) - num_active

    if (num_inactive > num_active and num_active > 0):
      self.scale_pos_weight = num_inactive/num_active
    else:
      self.scale_pos_weight = 1.0

  '''
  ==================================================================
  Functions for stitching together the individual features CSV files
  to create the training and validation sets.
  ==================================================================
  '''

  # Write feature imprtances to file
  def __importance_to_csv(
    self,
    run_id,
    df_dimension_reduction_importances,
    combined_model,
    drugs_model,
    proteins_model,
    df_features,
    df_drugs,
    df_proteins):

    results = []

    df_drug_importance = self.__get_features(drugs_model, df_drugs)
    df_protein_importance = self.__get_features(proteins_model, df_proteins)
    df_combined_importance = self.__get_features(combined_model, df_features)

    df_drug_importance.set_index('feature', inplace=True)
    df_protein_importance.set_index('feature', inplace=True)
    df_combined_importance.set_index('feature', inplace=True)

    for index, row in df_dimension_reduction_importances.iterrows():
      result = []
      feature = row['feature']
      result.append(run_id)
      result.append(feature)
      result.append(row['importance'])

      if feature in df_combined_importance.index:
        result.append(df_combined_importance.loc[feature]['importance'])
      else:
        result.append(0.0)

      if feature in df_drug_importance.index:
        result.append(df_drug_importance.loc[feature]['importance'])
      else:
        result.append(0.0)

      if feature in df_protein_importance.index:
        result.append(df_protein_importance.loc[feature]['importance'])
      else:
        result.append(0.0)

      results.append(result)

    df = pd.DataFrame(results, columns=[
      'run_id',
      'feature',
      'dim_red importance',
      'combined_classification',
      'drug_classification',
      'protein_classification'
      ])

    path = 'results/'+self.job_name+'/feature_importances_{}.csv'.format(run_id)
    df.to_csv(path, index=False)
    del df


  # Write parameters and metrics to CSV for each run
  def __params_and_metrics_to_csv(
    self,
    activity_thresholds,
    used_dimension_reduction_weights,
    used_training_weights,
    combined_results,
    drugs_results,
    proteins_results,
    roc_results):

    results = []

    for i in range(len(activity_thresholds)):
      result = []
      result.append(i)
      result.append(activity_thresholds[i])
      result.append(used_dimension_reduction_weights)
      result.append(used_training_weights)

      result.append(roc_results[i])

      result.append(combined_results[i]['accuracy'])
      result.append(combined_results[i]['precision'])
      result.append(combined_results[i]['recall'])
      result.append(combined_results[i]['f1_score_weighted'])
      result.append(combined_results[i]['f1_score_unweighted'])

      result.append(self.scale_pos_weight)
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
      'weighted_roc_auc',
      'accuracy',
      'precision',
      'recall',
      'f1_score_weighted',
      'f1_score_unweighted',
      'xgb_scale_pos_weight',
      'xgb_learning_rate',
      'xgb_n_estimators',
      'xgb_objective',
      'xgb_subsample',
      'xgb_min_child_weight',
      'xgb_max_depth',
      'xgb_gamma',
      'xgb_colsample_bytree'
      ])

    path = 'results/'+self.job_name+'/params_and_metrics.csv'
    df.to_csv(path, index=False)
    del df


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

    logging.debug('validation observations:')
    logging.debug(df_validation.head())

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
      result.append(row['activity_score'])
      result.append(validation_weights[index])

      cid_only_probability = drugs_model_results['probabilities'][index][1]
      pid_only_probability = proteins_model_results['probabilities'][index][1]
      combined_probability = combined_model_results['probabilities'][index][1]

      result.append(cid_only_probability)
      result.append(pid_only_probability)
      result.append(combined_probability)

      results.append(result)

    df = pd.DataFrame(results, columns=[
      'run_id',
      'run_threshold',
      'used_dim_red_weights',
      'used_training_weights',
      'cid',
      'pid',
      'activity',
      'activity_score',
      'validation_weight',
      'cid_only_predict_proba',
      'pid_only_predict_proba',
      'combined_predict_proba'
      ])

    path = 'results/'+self.job_name
    try:
      os.makedirs(path, exist_ok=True)
    except OSError:
      logging.debug ("Creation of the directory %s failed" % path)
    else:
      logging.debug ("Successfully created the directory %s " % path)

    file_path = 'results/'+self.job_name+'/results_{}_.csv'.format(run_id)
    df.to_csv(file_path, index=False)
    del df


  # load a specific features CSV file
  def __load_data(self, path, data_type=None):
    if data_type:
        df = pd.read_csv(path, index_col=0, dtype=data_type)
    else:
        df = pd.read_csv(path, index_col=0)
    logging.debug('Number of rows: {:,}'.format(len(df)))
    logging.debug('Number of columns: {:,}'.format(len(df.columns)))
    
    columns_missing_values = df.columns[df.isnull().any()].tolist()
    logging.debug('{} columns with missing values'.format(len(columns_missing_values)))
    
    logging.debug(df.head(2))
    
    return df

  # print out summary after each features merge
  def __print_merge_details(self, df_merge_result, df1_name, df2_name):
    logging.debug('Joining {} on protein {} yields {:,} rows and {:,} columns'. \
          format(df1_name, df2_name, len(df_merge_result), 
          len(df_merge_result.columns)))

  # Get the individual feature sets as data frames
  def __load_feature_files(self):
    logging.debug('===============================================')
    logging.debug('dragon_features.csv')
    logging.debug('===============================================')
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
    logging.debug('{} columns with missing values.'.format(len(columns_missing_values)))

    # replace NaNs with column means
    df_dragon_features.fillna(df_dragon_features.mean(), inplace=True)

    columns_missing_values = df_dragon_features.columns[df_dragon_features.isnull().any()].tolist()
    logging.debug('{} columns with missing values (after imputing): {}'.format(len(columns_missing_values), 
                                                                       columns_missing_values))
    logging.debug('===============================================')
    logging.debug('fingerprints.csv')
    logging.debug('===============================================')
    df_fingerprints = self.__load_data(self.data_loc+'drug_features/fingerprints.csv')
    
    logging.debug('===============================================')
    logging.debug('binding_sites.csv')
    logging.debug('===============================================')
    df_binding_sites = self.__load_data(self.data_loc+'protein_features/binding_sites.csv')
    
    # Name the index to 'pid' to allow joining to other feaure files later.
    df_binding_sites.index.name = 'pid'
    
    logging.debug('===============================================')
    logging.debug('expasy.csv')
    logging.debug('===============================================')
    df_expasy = self.__load_data(self.data_loc+'protein_features/expasy.csv')
    
    logging.debug('===============================================')
    logging.debug('profeat.csv')
    logging.debug('===============================================')
    df_profeat = self.__load_data(self.data_loc+'protein_features/profeat.csv')
    
    # Name the index to 'pid' to allow joining to other feaure files later.
    df_profeat.index.name = 'pid'
    
    # profeat has some missing values.
    s = df_profeat.isnull().sum(axis = 0)

    logging.debug('number of missing values for each column containing them is: {}'.format(len(s[s > 0])))

    # Drop the rows that have missing values.
    df_profeat.dropna(inplace=True)
    logging.debug('number of rows remaining, without NaNs: {:,}'.format(len(df_profeat)))
    
    return {'df_dragon_features': df_dragon_features,
           'df_fingerprints': df_fingerprints,
           'df_binding_sites': df_binding_sites,
           'df_expasy': df_expasy,
           'df_profeat': df_profeat}


  def __create_features(self, split_file_name, out_file_name, feature_sets):
    logging.debug('===============================================')
    logging.debug('interactions.csv')
    logging.debug('===============================================')
    
    # load interactions.csv
    df_interactions = self.__load_data(
      self.data_loc+'training_validation_split/' + split_file_name)

    logging.debug(df_interactions.head())
    
    # Get the individual feature sets
    df_dragon_features = feature_sets['df_dragon_features']
    df_fingerprints = feature_sets['df_fingerprints']
    df_binding_sites = feature_sets['df_binding_sites']
    df_expasy = feature_sets['df_expasy']
    df_profeat = feature_sets['df_profeat']
    
    logging.debug('===============================================')
    logging.debug('Join the data using {}'.format(split_file_name))
    logging.debug('===============================================')
    
    # Form the complete feature set by joining the data frames according to _cid_ and _pid_.
    # See the data readme in the Gitbug repository:
    # https://github.com/BrianDavisMath/FDA-COVID19/tree/master/data.
    
    # By convention, the file features should be concatenated in the following order (for consistency):
    # **binding_sites**, **expasy**, **profeat**, **dragon_features**, **fingerprints**.
    
    logging.debug('-----------------------------------------------')
    logging.debug('df_interactions + df_binding_sites = df_features ')
    df_features = pd.merge(df_interactions, df_binding_sites, on='pid', how='inner')
    self.__print_merge_details(df_features, 'interactions', 'binding_sites')
    
    logging.debug('-----------------------------------------------')
    logging.debug('df_features + df_expasy ')
    df_features = pd.merge(df_features, df_expasy, on='pid', how='inner')
    self.__print_merge_details(df_features, 'features', 'expasy')
    
    logging.debug('-----------------------------------------------')
    logging.debug('df_features + df_profeat ')
    df_features = pd.merge(df_features, df_profeat, on='pid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_profeat')
    
    logging.debug('-----------------------------------------------')
    logging.debug('df_features + df_dragon_features ')
    df_dragon_features.index.name = 'cid'
    df_features = pd.merge(df_features, df_dragon_features, on='cid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_dragon_features')
    
    logging.debug('-----------------------------------------------')
    logging.debug('df_features + df_fingerprints ')
    df_features = pd.merge(df_features, df_fingerprints, on='cid', how='inner')
    self.__print_merge_details(df_features, 'features', 'df_fingerprints')
    
    logging.debug('-----------------------------------------------')
    logging.debug('Number of rows in joined feature set: {:,}'.format(len(df_features)))
    logging.debug('Number of columns in joined feature set: {:,}'.format(len(df_features.columns)))
    
    # release memory used by previous dataframes.
    del df_interactions
    del df_binding_sites
    del df_expasy
    del df_profeat
    del df_dragon_features
    del df_fingerprints
    
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

    df = df_features.loc[:, ~df_features.columns.isin(self.non_feature_columns)]
    top_feature_cols = df.columns.values[feature_indices] # turn numbers back into column names

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
    logging.debug('{}: columns: {:,}'.format(file, len(columns)))
    return columns

  # load test data and gather metrics
  def __gather_metrics(self, df_in, model, df_validation):
    # Take only the reduced set of columns.
    df_validation = df_validation[df_in.columns.tolist()]

    logging.debug('rows: {:,}, columns: {:,}'.format(len(df_validation), len(df_validation.columns)))

    # weighted F1 score
    training_features_active = df_in[df_in['activity']==1]
    training_features_inactive = df_in[df_in['activity']==0]
    
    training_features_active = \
      training_features_active.loc[:, ~training_features_active.columns.isin(self.non_feature_columns)]
    training_features_inactive = \
      training_features_inactive.loc[:, ~training_features_inactive.columns.isin(self.non_feature_columns)]

    validation_features_active = df_validation[df_validation['activity']==1]
    validation_features_inactive = df_validation[df_validation['activity']==0]

    # Needed later for reporting
    df_validation_features = validation_features_active.append(validation_features_inactive)

    y_test = df_validation_features['activity'].values

    # remove the non-feature columns
    validation_features_active = \
      validation_features_active.loc[:, ~validation_features_active.columns.isin(self.non_feature_columns)]

    validation_features_inactive = \
      validation_features_inactive.loc[:, ~validation_features_inactive.columns.isin(self.non_feature_columns)]
    
    X_test = validation_features_active.append(validation_features_inactive)

    # accuracy, precision and recall
    X_test.columns = list(range(0, len(X_test.columns)))

    # make predictions for test data
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    accuracy = sum(np.diag(cm))/cm.sum()*100
    precision = cm[1][1]/sum(cm[:, 1])*100
    recall = cm[1][1]/sum(cm[1, :])*100

    logging.debug("Accuracy = {:0.2f}%".format(accuracy))
    logging.debug("Precision = {:0.2f}%".format(precision))
    logging.debug("Recall = {:0.2f}%".format(recall))

    active_weights, inactive_weights = self.__get_validation_weights(
        training_features_active, 
        training_features_inactive,
        validation_features_active, 
        validation_features_inactive)

    assert(len(inactive_weights) + len(active_weights) == len(df_validation))

    weights = list(active_weights) + list(inactive_weights)

    f1score = f1_score(y_test, y_pred, average='binary', sample_weight=weights)

    logging.debug('F1 Score (weighted): {:0.2f}%'.format(f1score*100))
    
    f1score_unweighted = f1_score(y_test, y_pred, average='binary')
    logging.debug('F1 Score (unweighted): {:0.2f}%'.format(f1score_unweighted*100))
    
    probabilities = model.predict_proba(X_test)

    return {
      'model': model,
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
    logging.debug('{:,} drug featues'.format(len(cid_features)))
    return cid_features


  # load protein data column names
  def __get_protein_column_names(self):
    pid_binding_sites = self.__get_csv_feature_names('protein_features/binding_sites.csv')
    pid_expasy = self.__get_csv_feature_names('protein_features/expasy.csv')
    pid_profeat = self.__get_csv_feature_names('protein_features/profeat.csv')

    pid_features = pid_binding_sites + pid_expasy + pid_profeat
    logging.debug('{:,} protein featues'.format(len(pid_features)))
    return pid_features

  def __get_xgb(self):
    xgb = XGBClassifier(scale_pos_weight=self.scale_pos_weight,
                        learning_rate=self.learning_rate, 
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
    
    X = df_in.loc[:, ~df_in.columns.isin(self.non_feature_columns)]
    X.columns = list(range(0, len(X.columns)))

    xgb = self.__get_xgb()

    sample_weight = None
    if use_weights == True:
      #sample_weight = df_in['activity_score']
      sample_weight = self.training_weights

    model = self.__train(X, Y, xgb=xgb, sample_weight=sample_weight)
    training_probabilities = model.predict_proba(X)
    del X
    del Y

    # load test data and gather metrics
    results = self.__gather_metrics(df_in, model, df_validation)
    results['training_probabilities'] = training_probabilities
    return results


def main(argv):
  training_features = None
  validation_features = None
  max_activity_threshold = None
  activity_threshold_step = None
  activity_threshold_stop = 0.0
  data_folder = 'data/'
  use_dimension_reduction_weights = 'True'
  use_training_weights = 'True'
  job_name = ''

  
  # check arguments.
  try:
    opts, args = getopt.getopt(argv,"ha:s:m:f:d:t:n:",["athresh=", "step=", "stop=", "data=", "dweights=", "tweights=", "name="])
  except getopt.GetoptError:
    print('\n\n')
    logging.debug('job.py -a <max_activity_threshold> -s <activity_threshold_step> -m <activity_threshold_stop> \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights> -n <job_name>')
    print('\n\n')
    sys.exit(2)

  if len(opts) < 2:
    print('\n\n')
    logging.debug('job.py -a <max_activity_threshold> -s <activity_threshold_step> -m <activity_threshold_stop>  \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights> -n <job_name>')
    print('\n\n')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('\n\n')
      logging.debug('job.py -a <max_activity_threshold> -s <activity_threshold_step> -m <activity_threshold_stop>  \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights> -n <job_name>')
      print('\n\n')
      sys.exit()
    elif opt in ("-a", "--athresh"):
      max_activity_threshold = arg
    elif opt in ("-s", "--step"):
      activity_threshold_step = arg
    elif opt in ("-m", "--stop"):
      activity_threshold_stop = arg
    elif opt in ("-f", "--data"):
      data_folder = arg
    elif opt in ("-d", "--dweights"):
      use_dimension_reduction_weights = arg
    elif opt in ("-t", "--tweights"):
      use_training_weights = arg
    elif opt in ("-n", "--name"):
      job_name = arg


  if max_activity_threshold is None or activity_threshold_step is None:
    print('\n\n')
    logging.debug('job.py -a <max_activity_threshold> -s <activity_threshold_step> -m <activity_threshold_stop>  \
-f <data_folder> -d <use_weights_for_dimension_reduction> -t <use_training_weights> -n <job_name>')
    print('\n\n')
    sys.exit()

  logging.debug('max_activity_threshold is {}'.format(max_activity_threshold))
  logging.debug('activity_threshold_step is {}'.format(activity_threshold_step))
  logging.debug('activity_threshold_stop is {}'.format(activity_threshold_stop))
  logging.debug('use_dimension_reduction_weights is {}'.format(use_dimension_reduction_weights=='True'))
  logging.debug('use_training_weights is {}'.format(use_training_weights=='True'))

  xgb = XGBoostClassifier(
    max_activity_threshold=max_activity_threshold,
    activity_threshold_step=activity_threshold_step,
    activity_threshold_stop=activity_threshold_stop,
    data_folder=data_folder, 
    use_dimension_reduction_weights=use_dimension_reduction_weights=='True',
    use_training_weights=use_training_weights=='True',
    job_name=job_name)


if __name__== "__main__":
  main(sys.argv[1:])




