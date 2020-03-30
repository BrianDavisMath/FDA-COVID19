# Disclaimer / Usage

The data set contained in this file is not yet ready for public release, as it requires more analysis. The data set is approved only for the limited purpose of making binding predictions for FDA approved drugs against
the proteins associated to the novel coronavirus (SARS-CoV-2).

# Status / Versioning

protein_features/binding_sites.csv in this version (v0) of the data set IS INCOMPLETE, and the FDA_drug_features folder is empty. As of this writing, protein_features/binding_sites.csv contains binding site features for only approximately 25% of the proteins (PID's) in the full set. All FDA approved drugs should already appear as SOME cid in the dataset, but I have yet to identify them. A complete data set will be released as version v1 (expected to be available by end-of-day Monday, 3/30/20).

Here's a link to v0:
https://drive.google.com/file/d/1DDp2fZpO43DxcpiYlqbC_jKuavp4uHHO/view?usp=sharing

# Project

The goal of this project is to construct a model which predicts which FDA approved drugs (if any) will bind to proteins
of SARS-CoV-2. The data set consists of binary binding labels for pairs of drug-like molecules (drugs) and proteins.
Some potentially relevant features have been precomputed. They have not been normalized,
                        ** with the exception of binding_sites.csv **.

This data set contains:
    interactions.csv
        Each row contains a drug / protein pair (cid / pid) and a binary label indicating if the pair of molecules bind.
    example_feature_concatenation.csv
        Consists of the feature vector of a single cid-pid pair. It serves as an exemplar for concatenating
        the feature vectors correctly. Beware that the feature files may have duplicated column names, but that the
        associated values may be different. The cid - pid pair is the first row of interactions.csv: 38258,CAA96025
    protein_features
        Each csv file in this folder contains rows indexed by a protein identifier (pid).
    drug_features
        Each csv file in this folder contains rows indexed by a drug identifier (cid).
    coronavirus_features
        Each csv file in this folder contains rows indexed by a protein identifier (pid).
    FDA_drug_features
        Each csv file in this folder contains rows indexed by a drug identifier (cid).

The files contained in FDA_drug_features contain the subset of rows of the corresponding file in drug_features which
correspond to FDA approved drugs. The files contained in coronavirus_features are not expected to be duplicates of rows
in the corresponding files in the protein_features folder.

## Consolidated Features

The full (raw) feature set for a (cid, pid)-pair is a concatenation of the feature vectors from each relevant file:
The row indexed by the particular cid in each file in the drug_features folder and the row indexed by the particular pid in 
each file in the drug_features folder.

By convention, the file features should be concatenated in the following order (for consistency):
	binding_sites, expasy, profeat, dragon_features, mordered_features, fingerprints

Use the file example_feature_concatenation.csv to verify that you are matching the convention.

