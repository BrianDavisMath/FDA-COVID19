# Jupyter notebooks workflow

There's a loose workflow for using the notebooks in this folder for conditioning the data and subsequently exploring through modeling.


1. [name]-features.ipynb - stitches the interactions, drug and protein CSV files together into one large set

1. centroid-sampling - sample data from the full set to get a balanced representation of drugs, proteins and activation class. This notebook also drops columns that have zero variance

1. dim-red-via-correlation.ipynb - drop highly correlated columns

1. your choice of notebook for modeling, e.g. _greg-dimension-reduction-xgboost.ipynb_ or _greg-dimension-reduction-genetic.ipynb_ to find a smaller set of candidate features for subsequent modeling
