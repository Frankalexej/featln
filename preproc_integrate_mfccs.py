import pandas as pd
import torch
import os
from paths import *

def intergate_mfcc(guide_file, src_): 
    guide_df = pd.read_csv(guide_file)
    rec_list = guide_df["rec"].unique().tolist()
    output_tensor = torch.empty(0, 25, 39)
    for rec_name in rec_list: 
        output_tensor = torch.cat([output_tensor, torch.load(os.path.join(src_, f"{rec_name}.mfcc"))], dim=0)
        print(rec_name)
    return output_tensor

"""
['phone_anno_train.csv', 
                    'phone_random_train.csv', 
                    'phone_random_test.csv', 
                    'phone_anno_test.csv', 
                    'phone_anno_validation.csv', 
                    'phone_random_validation.csv']
"""
if __name__ == "__main__":
    for logname in ['phone_anno_train.csv', 
                    'phone_random_train.csv', 
                    'phone_random_test.csv', 
                    'phone_anno_test.csv', 
                    'phone_anno_validation.csv', 
                    'phone_random_validation.csv']: 
        src_ = phone_seg_random_MF_path if "random" in logname else phone_seg_anno_MF_path
        mfcc_tensor = intergate_mfcc(os.path.join(bsc_use_path, logname), src_)
        mfcc_filename = logname.split(".")[0]
        torch.save(mfcc_tensor, os.path.join(bsc_use_path, f"{mfcc_filename}.mfcc"))
    # mfcc_tensor = intergate_mfcc(os.path.join(phone_seg_anno_log_path, "log.csv"), phone_seg_anno_path)
    # torch.save(mfcc_tensor, os.path.join(bsc_path, "phone_anno.mfcc"))