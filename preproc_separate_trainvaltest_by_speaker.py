import pandas as pd
import torch
import os
import numpy as np
from paths import *

def separate_TVT(guide_file): 
    guide_df = pd.read_csv(guide_file)
    speakers = guide_df["rec"].str.slice(stop=3).unique().tolist()
    np.random.shuffle(speakers)
    num_speakers = len(speakers)
    ratios = [0.8, 0.1, 0.1]
    train_speakers = speakers[:int(num_speakers*ratios[0])]
    val_speakers = speakers[int(num_speakers*ratios[0]):int(num_speakers*(ratios[0] + ratios[1]))]
    test_speakers = speakers[int(num_speakers*(ratios[0] + ratios[1])):]
    
    # create output tensors for each set of speakers
    train_df = guide_df[guide_df["rec"].str.slice(stop=3).isin(train_speakers)]
    val_df = guide_df[guide_df["rec"].str.slice(stop=3).isin(val_speakers)]
    test_df = guide_df[guide_df["rec"].str.slice(stop=3).isin(test_speakers)]
    
    return train_df, val_df, test_df


if __name__ == "__main__": 
    prefix = "phone_random_"
    tr, va, te = separate_TVT(os.path.join(phone_seg_random_log_path, "log.csv"))
    tr.to_csv(os.path.join(bsc_use_path, prefix + "train.csv"), index=False)
    va.to_csv(os.path.join(bsc_use_path, prefix + "validation.csv"), index=False)
    te.to_csv(os.path.join(bsc_use_path, prefix + "test.csv"), index=False)