# FDA-COVID19
A repo for the FDA repurposing project (Joint w/ Markey Cancer Center and Oak Ridge National Labs)

## Workflow
The data aggregation process will be documented elsewhere. For the purposes of this work, we will assume that we have received a zipped folder containing:
  * interactions.txt: Each row contains a CID (drug/ligand), a PID (protein), and a binary label.
  * fda_drug_cids.csv: each row contains the cid of an FDA-approved (or experimental) drug.
  * a folder protein_features: each file in the folder contains a row of features for each CID.
  * a folder ligand_features: each file in the folder contains a row of features for each PID.
  * a folder coronavirus_features: each file in the folder contains a row of features for each PID.

### Assembling Raw Features
  * A csv file indexed by PID is created by concatenating the rows of each file in protein_features folder.
  * A csv file indexed by CID is created by concatenating the rows of each file in ligand_features folder.

### Dimension Reduction / Feature Selection
  * The raw feature dimensionality is too high, so we will reduce the total feature dimension for each interaction example to 1,000.
  * An evaluator model will be created to aide in prototyping different algorithms and parameter settings for feature selection / dimension reduction.
  * The baseline feature selection method will be the random_forest_feature_selector.
   
### Prediction Model
  * We will compare the performance of two prediction models: 
      1. A fully connected feedforward neural network 
      2. XGBoost
  * On each training run, it will be fed with features from a feature selection model.
