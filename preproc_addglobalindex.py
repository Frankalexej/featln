import pandas as pd
import os
from paths import *

def add_global_index(path):
    # read in the csv file
    df = pd.read_csv(path)

    # add a global index column
    # if there is such column, just overwrite it
    df['global_idx'] = range(0, len(df))

    # write the updated dataframe back to the csv file
    df.to_csv(path, index=False)

if __name__ == "__main__":
    add_global_index(os.path.join(phone_seg_anno_log_path, "log.csv"))
    add_global_index(os.path.join(phone_seg_random_log_path, "log.csv"))
