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
        # print(f"-----------------{file}------------------")
        try: 
            wave, sr = torchaudio.load(os.path.join(src_dir, file))

            single_mfcc_feats = transformer(wave)
            # resampled_wave = torch.tensor(signal.resample(wave, 4240, axis=1))
            single_mfcc_feats_resampled = torch.tensor(signal.resample(single_mfcc_feats, 25))
            mfcc_feats = torch.cat([mfcc_feats, single_mfcc_feats_resampled.unsqueeze(0)], dim=0)
        except Exception as e: 
            print(e)
    
    torch.save(mfcc_feats, os.path.join(tgt_dir, f"{save_name}.mfcc"))
    print(save_name)

def generate_dict(df):
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
    # src_ = phone_seg_random_rec_path
    # tgt_ = phone_seg_random_MF_path
    # log_ = os.path.join(bsc_path, "random-log.csv")
    src_ = phone_seg_anno_rec_path
    tgt_ = phone_seg_anno_MF_path
    log_ = os.path.join(bsc_path, "anno-log.csv")
    guide_log = pd.read_csv(log_)
    guide_log = guide_log[guide_log['n_frames'] > 400]
    guide_log = guide_log[guide_log['duration'] <= 2.0]

    guide_log.to_csv(log_, index=False)
    guide_log = pd.read_csv(log_)

    workmap = generate_dict(guide_log)
    worklist = sorted(workmap.keys())
    divided_worklist = divide_work(worklist, multiprocessing.cpu_count())
    for workchunk in divided_worklist: 
        pool = multiprocessing.Pool(processes=32)

        for i, rec in enumerate(workchunk):
            print(f"Start {rec}")
            files = workmap[rec]
            filelist = [f"{rec}_{str(idx).zfill(8)}.wav" for idx in files]
            result = pool.apply_async(process_files, args=(src_, tgt_, filelist, rec))
        pool.close()
        pool.join()