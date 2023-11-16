import torchaudio
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import signal
import multiprocessing
import pandas as pd

from paths import *
from preproc_mfccTransform import MFCCTransform
from misc_progress_bar import draw_progress_bar


transformer = MFCCTransform()

def process_files(src_dir, tgt_dir, files, save_name):
    mfcc_feats = torch.empty(0, 25, 39)
    for file in files:
        try: 
            wave, sr = torchaudio.load(os.path.join(src_dir, file))
            resampled_wave = torch.tensor(signal.resample(wave, 4240, axis=1))
            mfcc_feats = torch.cat([mfcc_feats, transformer(resampled_wave).unsqueeze(0)], dim=0)
        except Exception as e: 
            print(e)
    
    torch.save(mfcc_feats, os.path.join(tgt_dir, f"{save_name}.mfcc"))
    print(save_name)

def generate_dict(csv_path):
    # Read in the CSV file as a pandas dataframe
    df = pd.read_csv(csv_path)
    rec_dict = df.groupby('rec').groups
    idx_list = df["idx"].tolist()
    # Sort the lists of indices for each 'rec' value
    for rec in rec_dict:
        rec_dict[rec] = sorted([idx_list[idx] for idx in rec_dict[rec]])
    return rec_dict

def divide_work(worklist, n):
    chunks = []
    for i in range(0, len(worklist), n):
        chunks.append(worklist[i:i+n])
    return chunks



if __name__ == '__main__':
    src_ = phone_seg_anno_rec_path
    tgt_ = phone_seg_anno_path

    workmap = generate_dict(os.path.join(phone_seg_anno_log_path, "log.csv"))
    worklist = sorted(workmap.keys())
    for rec in worklist: 
        files = workmap[rec]
        filelist = [f"{rec}_{str(idx).zfill(8)}.wav" for idx in files]
        process_files(src_, tgt_, filelist, rec)