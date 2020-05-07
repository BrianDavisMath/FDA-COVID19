# Jupyter notebooks workflow

There's a loose workflow for using the notebooks in this folder for conditioning the data and subsequently exploring through modeling.


1. _greg-features.ipynb_ - stitches the interactions, drug and protein CSV files together into one large set

1. _greg-dimension-reduction-xgboost.ipynb_ — consists of a number of experiments for using XGBoost for dimension reduction and then subsequently for training the final model. The latter experiments begin to converge on our solution.
