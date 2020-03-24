# FDA-COVID19
A repo for the FDA repurposing project (Joint w/ Markey Cancer Center and Oak Ridge National Labs)

## Workflow
The data aggregation process will be documented elsewhere. For the purposes of this work, we will assume that we have received a zipped folder containing:
  * interactions.txt: Each row contains a CID (drug/ligand), a PID (protein), and a binary label.
  * a folder protein_features: each file in the folder contains a row of features for each CID.
  * a folder ligand_features: each file in the folder contains a row of features for each PID.
 
### Assembling Raw Features
  * A csv file indexed by PID is created by concatenating the rows of each file in protein_features folder.
  * A csv file indexed by CID is created by concatenating the rows of each file in ligand_features folder.

### Training / Validation / Test Split
  * A collection of proteins (PID's) and ligands (CID's) will be selected, and all interactions involving these CID/PID will be reserved for the test set.
  * The remaining interaction examples will be split randomly 80/20 for the training and validation sets, respectively.

### Dimension Reduction / Feature Selection
  * The raw feature dimensionality is too high, so we will reduce the total feature dimension for each interaction example to 1,000.
  * An evaluator model will be created to aide in prototyping different algorithms and parameter settings for feature selection / dimension reduction.
  * The baseline feature selection method will be the random_forest_feature_selector.
  
### Spatial Statistics
  * Using the preliminary embedding produced by the random_forest_feature_selector, a weight between zero and one will be assigned to each example interaction in the validation set. This weighting will be used in computing weighted performance metrics, which penalize "perceived" model overfitting.
  
### Prediction Model
  * A fully connected feedforward neural network will act as the final prediction model.
  * On each training run, it will be fed with features from a feature selection model.
  * It will record weighted and unweighted loss functions for the training and validation sets, and terminate training when the rates-of-change of the weighted and unweighted validation loss diverge.
