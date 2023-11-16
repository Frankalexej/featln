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


if __name__ == "__main__":
    # mfcc_tensor = intergate_mfcc(os.path.join(phone_seg_anno_log_path, "log.csv"), phone_seg_anno_path)
    # torch.save(mfcc_tensor, os.path.join(bsc_path, "phone_anno.mfcc"))

    mfcc_tensor = intergate_mfcc(os.path.join(phone_seg_random_log_path, "log.csv"), phone_seg_random_path)
    torch.save(mfcc_tensor, os.path.join(bsc_path, "phone_random.mfcc"))