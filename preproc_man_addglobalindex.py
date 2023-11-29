import pandas as pd
import os
from paths import *
from ssd_paths import *

def add_global_index(path):
    # read in the csv file
    df = pd.read_csv(path)

    # add a global index column
    # if there is such column, just overwrite it
    df['global_idx'] = range(0, len(df))

    # write the updated dataframe back to the csv file
    df.to_csv(path, index=False)
"""
'phone_random_train.csv', 
'phone_random_test.csv', 
'phone_random_validation.csv'
"""
if __name__ == "__main__":
    for logname in [
                    'phone_anno_test.csv', 
                    'phone_anno_validation.csv', 
                    ]: 
        add_global_index(os.path.join(as_use_path, logname))
