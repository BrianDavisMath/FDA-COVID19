import pandas as pd

# an example feature selector. Requires two things:
#   - a name attribute
#   - a call function that takes 
class IdentityFeatureSelector(object):
    def __init__(self):
        self.name = "identity"
        return

    def __call__(self, raw_data_path, processed_data_path):
        # load data
        raw_data = pd.read_csv(raw_data_path)
        
        # you would do somthing with it here

        # save to csv
        raw_data.to_csv(processed_data_path)
        return
    