# XGBoost Modeling Runner

This folder contains the scripts required to run multiple passes on the XGBoost-based dimension reduction and subsequent drug-receptor binding classification.


## job.py and data folder

The _[job.py](job.py)_ python script contains the dimension reduction and classification code. It expects a _data_ folder containing the features files as follows:

```
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
```

The above files are joined together to create the training and validation sets using the _cid_ and _pid_ values contained in _training_interactions.csv_ and _validation_interactions.csv_. The actual training set that is used in a single pass of the modeling is determined by a subset selected using the _[-a]_ and _[-s]_ parameters (see below).


## Parameters

* **[-a] max_activity_threshold** - used to sub-sample the training data for the first run

* **[-s] activity_threshold_step** - the amount to reduced the activity threshold on each run before sub-sampling the training data

* **[-f] data_folder** - defaults to "/data". Specifies the location of the data from which the features will be selected.

* **[-d] use_dimension_reduction_weights** - [True|False] whether to use sample_activity_score for sample_weight when training XGBoost for feature selection

* **[-t] use_training_weights** - [True|False] whether to use sample_activity_score for sample_weight when training XGBoost for activity classification

* **[-n] name** - job name used to place results in a directory of that name


## Activity score threshold

The _[-a]_ and _[-s]_ parameters determine how the training data are sampled and how many modeling runs will occur. For example _-a 0.03 -s 0.01_ will first create a sample based upon the 0.03 threshold then one for 0.02 then finally one for 0.01. The dimension reduction and classification will then be carried out on each sample. The output files will be numbered according to each run; 0, 1 and 2 in this instance, since there are three samples defined.


## Results and outputs

All results are writen to a dynamically created _results_ folder. If a _name (-n)_ parameter is passed to the job.py script then a subdirectory of that name will created. The following files are written:

* **results_[run_id].csv** - model outputs including binary classification for activity as well as probability of activity for each cid/pid pair in the validation set;

* **feature_importances_[run_id].csv** - the set of most important features, along with their information gain values, as returned by the first run of XGBoost for dimension reduction as well as for each classification (cid-only, pid-only and combined).

* **params_and_metrics.csv** - the accuracy, precision, recall, weighted and unweighted F1 score representing model performane along with the XGBoost parameters used


## Run the script

The script can be executed using the _run.sh_ bash script in this folder or it can be run directly, e.g.:

```bash
python job.py -a 0.02 -s 0.01 -f test_data/ -d False -t False -n 'test_0'
```


## Test data

The _test_data_ folder contains very reduced versions of the expected input data. This enables very quick test runs of the job.py script and to test updates quickly.


## Python requirements

See the _[requirements.txt](requirements.txt)_ file for the required Python modules.