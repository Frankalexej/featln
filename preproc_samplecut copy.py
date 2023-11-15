import torch
import torchaudio
import torchaudio.transforms as transforms
import os
import math
from multiprocessing import Pool, cpu_count
import sys
import torch.nn as nn
from scipy import signal

from paths import *
from mio import *
from sampler import *
from my_utils import *
from sound_proc import *
from misc_progress_bar import draw_progress_bar

"""
The logic here is: 
Put all sounds into one big chunk, and 
"""
# define transformer
class MFCCTransform(nn.Module): 
    def __init__(self): 
        super().__init__()
        # self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=64)
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        # self.transform = torchaudio.transforms.MFCC(n_mfcc=13)
    
    def forward(self, waveform, sr=16000): 
        # extract mfcc
        feature = torchaudio.compliance.kaldi.mfcc(waveform, sample_frequency=sr)

        # add deltas
        d1 = torchaudio.functional.compute_deltas(feature)
        d2 = torchaudio.functional.compute_deltas(d1)
        feature = torch.cat([feature, d1, d2], dim=-1)

        # Apply normalization (CMVN)
        eps = 1e-9
        mean = feature.mean(0, keepdim=True)
        std = feature.std(0, keepdim=True, unbiased=False)
        feature = (feature - mean) / (std + eps)
        return feature

# load distribution parameters
params = load_gamma_params("phones_length_gamma.param")
transform = MFCCTransform()

## Define Open and Cut Functions
# ground truth cut
def open_and_cut(wave_path, annos_path, params, no_real_cut=False):
    filtered_df = filter_tokens_and_get_df(annos_path, keepSIL=False)
    flat_starts, flat_ends, c_duration = filtered_df["start_time"].to_numpy(), filtered_df["end_time"].to_numpy(), filtered_df["duration"].to_numpy()

    if "theory_segments" in filtered_df.columns:
        c_thsegs = filtered_df["theory_segments"].to_numpy()
    else: 
        c_thsegs = ""

    if "produced_segments" in filtered_df.columns:
        c_prsegs = filtered_df["produced_segments"].to_numpy()
    else: 
        c_prsegs = ""

    if not no_real_cut: 
        sp = Sound_Proc()
    
        rec, sample_rate = torchaudio.load(wave_path)

        cut_recs = sp.cut_rec(rec, flat_starts, flat_ends)

        # NOTE: This is added because a very small proportion of the data are strangely having zero n_frames (which I don't know yet why)
        # to filter them out, I added this n_frames
        cut_n_frames = [cut_rec.shape[1] for cut_rec in cut_recs]
        cut_n_frames = np.array(cut_n_frames)
    else: 
        pass
        # ref_anno_df = pd.read_csv(os.path.join(word_seg_anno_log_ref_path, os.path.basename(annos_path)))
        # assert ref_anno_df.shape[0] == filtered_df.shape[0]
        # cut_n_frames = ref_anno_df["n_frames"].to_numpy()
        # cut_recs = []
    
    tokens = filtered_df["token"].to_numpy()
    
    cst, cet = flat_starts, flat_ends
    
    
    # Framify
    # Create a dictionary with the three lists as values and the column names as keys
    data = {'rec': os.path.splitext(os.path.basename(wave_path))[0], "idx": list(map("{:08d}".format, range(len(c_duration)))), 'start_time': cst, 'end_time': cet, 'token': tokens, 'duration': c_duration, 'n_frames':cut_n_frames, 'theory_segments': c_thsegs, 'produced_segments': c_prsegs}
    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    return cut_recs, df

# Random Sampling Cut
def open_and_cut_phones_random_sampling(wave_path, anno_path, params, no_real_cut=False): 
    sp = Sound_Proc()
    metadata = torchaudio.info(wave_path)
    rec_len = sp.get_rec_length(metadata)
    samples = gamma_samples_sum(rec_len, params, shift=0.0125)

    flat_starts, flat_ends = samples2idx_with_se(samples)
    
    rec, sample_rate = torchaudio.load(wave_path)
    cut_recs = sp.cut_rec(rec, flat_starts, flat_ends)

    cut_n_frames = [cut_rec.shape[1] for cut_rec in cut_recs]
    cut_n_frames = np.array(cut_n_frames)
    
    cst, cet = flat_starts, flat_ends
    c_duration = [cet[i] - cst[i] for i in range(len(cst))]
    
    # Framify
    # Create a dictionary with the three lists as values and the column names as keys
    data = {'rec': os.path.splitext(os.path.basename(wave_path))[0], "idx": list(map("{:08d}".format, range(len(c_duration)))), 'start_time': cst, 'end_time': cet, 'token': "", 'duration': c_duration, 'n_frames':cut_n_frames, 'theory_segments': "", 'produced_segments': ""}
    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    return cut_recs, df


"""
### Multiprocessing
To make processing easier, both open-and-cut functions return the same output: `cut_recs` 
(a list of NumPy arrays) and a `token_list` (a Pandas DataFrame).

In order to speed up the processing time, you can use multiprocessing to plan the work and 
distribute it to the two open-and-cut functions. This will allow each function to work on a 
separate process, which can be run simultaneously, potentially reducing the overall processing time.
"""
def collaboration_single_work(my_work_pool, fun, my_wave_dir, my_anno_dir, my_save_dir, my_log_dir, my_params, no_real_cut): 
    print("Working from {} to {}".format(my_work_pool[0], my_work_pool[-1]))
    total_my_work = len(my_work_pool)
    for idx, rec_name in enumerate(my_work_pool): 
        draw_progress_bar(idx, total_my_work)
        rec_raw, ext = os.path.splitext(rec_name)
        cut_recs, corr_df = fun(
            os.path.join(my_wave_dir, rec_name), 
            os.path.join(my_anno_dir, rec_raw + ".csv"),
            my_params, 
            no_real_cut=no_real_cut
        )

        rec_name = corr_df["rec"][0]
        suffices = corr_df["idx"]
        for cut_idx, cut_seg in enumerate(cut_recs): 
            save_name = os.path.join(my_save_dir, f"{rec_name}_{suffices[cut_idx]}.mfcc")
            resampled_wave = torch.tensor(signal.resample(cut_seg, 4240, axis=1))
            mfcc_feats = transform(resampled_wave)
            torch.save(mfcc_feats, save_name)
        
        csv_path = os.path.join(my_log_dir, "f{rec_name}.csv")
        # write the combined DataFrame to the same CSV file
        corr_df.to_csv(csv_path, index=False)
    print("Work from {} to {} ends".format(my_work_pool[0], my_work_pool[-1]))

class MultiprocessManager: 
    def __init__(self, fun, my_wave_dir, my_anno_dir, my_save_dir, my_log_dir, my_params, no_real_cut=False, num_workers=4): 
        self.fun = fun
        self.my_wave_dir = my_wave_dir
        self.my_anno_dir = my_anno_dir
        self.my_save_dir = my_save_dir
        self.my_log_dir = my_log_dir
        self.my_params = my_params
        self.no_real_cut = no_real_cut
        self.num_workers = num_workers
    
    def divide_work(self, work):
        # determine the number of items per worker
        items_per_worker = math.ceil(len(work) / self.num_workers)

        # divide the work into chunks
        work_chunks = [work[i:i + items_per_worker] for i in range(0, len(work), items_per_worker)]

        return work_chunks
    
    def collaboration_work(self): 
        flat_tasks = os.listdir(self.my_wave_dir)
        task_pools = self.divide_work(flat_tasks)
        print(self.num_workers)
        p = Pool(self.num_workers)
        for i in range(self.num_workers):
            p.apply_async(collaboration_single_work, args=(task_pools[i], self.fun, self.my_wave_dir, self.my_anno_dir, self.my_save_dir, self.my_log_dir, self.my_params, self.no_real_cut))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

def csv_bind(log_dir): 
    # List all the CSV files in the directory that start with 's'
    directory = log_dir
    csv_files = sorted([f for f in os.listdir(directory) if f.startswith('s') and f.endswith('.csv')])

    # Read and concatenate the CSV files using pandas
    dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated dataframe as "log.csv"
    concatenated_df.to_csv(os.path.join(directory, 'log.csv'), index=False)

## Run 
# Random Sampling
def run(domain=None, no_real_cut=False): 
    print("Starting...")
    n_worker = cpu_count()
    if domain == "phone-random": 
        # random sampling
        mpm = MultiprocessManager(open_and_cut_phones_random_sampling, 
                                wav_path, phones_extract_path, 
                                phone_seg_random_path, 
                                phone_seg_random_log_path, 
                                params, num_workers=n_worker)
        
        mpm.collaboration_work()

        #### Bind csvs into one
        csv_bind(phone_seg_random_log_path)
    elif domain == "phone-anno": 
        # aligned cutting
        mpm = MultiprocessManager(open_and_cut, 
                                wav_path, 
                                phones_extract_path, 
                                phone_seg_anno_path, 
                                phone_seg_anno_log_path, 
                                params, no_real_cut=no_real_cut, 
                                num_workers=n_worker)
        
        mpm.collaboration_work()

        #### Bind csvs into one
        csv_bind(phone_seg_anno_log_path)


if __name__ == "__main__": 
    domain = None
    # Check if there are command line arguments
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        if len(sys.argv) > 2: 
            if sys.argv[2] == "no-real-cut": 
                no_real_cut = True
            else: 
                no_real_cut = False
    else:
        print("No command line arguments provided.")
    run(domain=domain, no_real_cut=no_real_cut)
    