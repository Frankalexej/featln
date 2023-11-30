import pandas as pd
import torch
import numpy as np
import shutil
import os
from paths import *
from ssd_paths import *
from misc_progress_bar import draw_progress_bar

def split_list(input_list, n):
    # Calculate the length of each sublist
    sublist_length = len(input_list) // n
    
    # Calculate the remainder for handling uneven division
    remainder = len(input_list) % n
    
    # Initialize the starting index for slicing
    start = 0
    
    # Iterate through the number of sublists
    for i in range(n):
        # Calculate the ending index for slicing
        end = start + sublist_length + (1 if i < remainder else 0)
        
        # Yield the sublist
        yield input_list[start:end]
        
        # Update the starting index for the next iteration
        start = end



def intergate_mfcc(workpool, src_, pool_idx = 0): 
    output_tensor = torch.empty(0, 25, 39)
    total = len(workpool)
    for idx, rec_name in enumerate(workpool): 
        output_tensor = torch.cat([output_tensor, torch.load(os.path.join(src_, f"{rec_name}.mfcc"))], dim=0)
        # print(rec_name)
        draw_progress_bar(idx + 1, total, content=f'\t{pool_idx}')
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
    for logname in ['phone_random_validation.csv', 'phone_random_test.csv']: 
        src_ = as_phone_seg_random_OMF_path if "random" in logname else as_phone_seg_anno_OMF_path
        # make a temp dir to store intermediate files
        temp_dir = os.path.join(ssd_aishell_path, "integrate_temp")
        mk(temp_dir)
        mfcc_filename = logname.split(".")[0]
        # src_ = as_phone_seg_anno_OMF_path
        guide_df = pd.read_csv(os.path.join(as_use_path, logname))
        rec_list = guide_df["rec"].unique().tolist()
        intermediate_workpools = list(split_list(rec_list, 3))
        for idx, workpool in enumerate(intermediate_workpools): 
            mfcc_tensor = intergate_mfcc(workpool, src_, pool_idx=idx)
            torch.save(mfcc_tensor, os.path.join(temp_dir, f"{mfcc_filename}-{idx:02d}.mfcc"))
        
        # integrate the intermediate files
        total_mfcc_tensor = torch.empty(0, 25, 39)
        total = len(intermediate_workpools)
        for idx in range(len(intermediate_workpools)): 
            total_mfcc_tensor = torch.cat([total_mfcc_tensor, torch.load(os.path.join(temp_dir, f"{mfcc_filename}-{idx:02d}.mfcc"))], dim=0)
            draw_progress_bar(idx + 1, total, content='Integrated')
        torch.save(total_mfcc_tensor, os.path.join(as_use_path, f"{mfcc_filename}.mfcc"))

        shutil.rmtree(temp_dir, ignore_errors=True)
    # mfcc_tensor = intergate_mfcc(os.path.join(phone_seg_anno_log_path, "log.csv"), phone_seg_anno_path)
    # torch.save(mfcc_tensor, os.path.join(bsc_path, "phone_anno.mfcc"))
    # python preproc_man_integrate_mfccs_stepbystep.py