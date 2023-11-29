import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import os
import torchaudio
from scipy import signal
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np
import pickle

class UngroundedSoundDataset(Dataset):
    """
    Loading ungrounded sound data.
    Filter out the ones with too few frames or those that are too long.

    Args:
        data_path (str): The path to the dataset. (torch.Tensor)
        guide_path (str): The path to the guide. (csv)

    Attributes:
        dataset (torch.Tensor): The loaded dataset.
        index_list (list): The list of indices of the dataset to be used.
    """
    def __init__(self, data_path, guide_path): 
        # Load the dataset
        self.dataset = torch.load(data_path)
        # Load the guide
        control_file = pd.read_csv(guide_path)
        # Filter out the ones with too few frames or those that are too long
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        assert len(control_file) == len(self.dataset)
        print(len(control_file), len(self.dataset))

        self.index_list = control_file['global_idx'].tolist()
    
    def __len__(self) :
        return len(self.index_list)
    
    def __getitem__(self, idx): 
        inp = self.dataset[idx]
        outp = inp[:, :13]
        return inp, outp
    
class GroundedSoundDataset(Dataset):
    """
    Loading ungrounded sound data.
    Filter out the ones with too few frames or those that are too long.

    Args:
        data_path (str): The path to the dataset. (torch.Tensor)
        guide_path (str): The path to the guide. (csv)

    Attributes:
        dataset (torch.Tensor): The loaded dataset.
        index_list (list): The list of indices of the dataset to be used.
    """
    def __init__(self, data_path, guide_path): 
        # Load the dataset
        self.dataset = torch.load(data_path)
        # Load the guide
        control_file = pd.read_csv(guide_path)
        # Filter out the ones with too few frames or those that are too long
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]

        self.index_list = control_file['global_idx'].tolist()
        self.token_list = control_file['token'].tolist()
    
    def __len__(self) :
        return len(self.index_list)
    
    def __getitem__(self, idx): 
        inp = self.dataset[idx]
        outp = inp[:, :13]
        token = self.token_list[idx]
        return inp, outp, token
    

class FlyingSoundDataset(Dataset):
    def __init__(self, data_path, guide_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.
        transform (Transform): when loading files, this will be applied to the sound data. 
        """
        control_file = pd.read_csv(guide_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.load_dir = data_path
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        return data, data[:, :13]

class FlyingGroundedSoundDataset(Dataset):
    def __init__(self, data_path, guide_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.
        transform (Transform): when loading files, this will be applied to the sound data. 
        """
        control_file = pd.read_csv(guide_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)
        token_col = control_file['token'].astype(str)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.tokenset = token_col.tolist()
        self.load_dir = data_path
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        return data, data[:, :13], self.tokenset[idx]

class MFCCTransform(nn.Module): 
    def __init__(self, sample_rate=16000, n_mfcc=13, normalizer=None): 
        super().__init__()
        self.normalizer = normalizer
        self.sample_rate = sample_rate
    
    def forward(self, waveform): 
        # extract mfcc

        feature = torchaudio.compliance.kaldi.mfcc(waveform, 
                                                   sample_frequency=self.sample_rate, 
                                                   dither=0, 
                                                   high_freq=8000)

        # add deltas
        d1 = torchaudio.functional.compute_deltas(feature)
        d2 = torchaudio.functional.compute_deltas(d1)
        feature = torch.cat([feature, d1, d2], dim=-1)

        # normalize
        if self.normalizer: 
            feature = self.normalizer(feature)

        return feature

class OldMFCCTransform(nn.Module): 
    def __init__(self, sample_rate=16000, normalizer=None): 
        super().__init__()
        self.normalizer = normalizer
        self.sample_rate = sample_rate
    
    def forward(self, waveform): 
        # extract mfcc

        waveform = waveform.squeeze(0).numpy()
        mfcc_feat = mfcc(waveform, self.sample_rate)
        delta_feat = delta(mfcc_feat, 2)
        delta_delta_feat = delta(delta_feat, 2)
        feature = np.concatenate((mfcc_feat, delta_feat, delta_delta_feat), axis=1)

        # normalize
        if self.normalizer: 
            feature = self.normalizer(feature)

        return feature



# class Resampler: 
#     @staticmethod
#     def resample_mfcc(mfcc, target_frame_num=25):
#         return torch.tensor(signal.resample(mfcc, target_frame_num, axis=0))
    
#     @staticmethod
#     def resample_wave(waveform, target_frame_num=4240):
#         return torch.tensor(signal.resample(waveform, target_frame_num, axis=1))
    
class Resampler(nn.Module): 
    def __init__(self, target_frame_num=25, axis=0): 
        super().__init__()
        self.target_frame_num = target_frame_num
        self.axis = axis
    
    def forward(self, mfcc): 
        return torch.tensor(signal.resample(mfcc, self.target_frame_num, axis=self.axis))


class Normalizer: 
    @staticmethod
    def norm_strip_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(1, keepdim=True)
        std = mel_spec.std(1, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_time_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec

    @staticmethod
    def norm_minmax(mel_spec):
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_strip_minmax(mel_spec):
        min_val = mel_spec.min(1, keepdim=True)[0]
        max_val = mel_spec.max(1, keepdim=True)[0]
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_pcen(mel_spec):
        return mel_spec
    

class DS_Tools:
    @ staticmethod
    def save_indices(filename, my_list):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(my_list, file)
            return True
        except Exception as e:
            print(f"An error occurred while saving the list: {e}")
            return False

    @ staticmethod    
    def read_indices(filename):
        try:
            with open(filename, 'rb') as file:
                my_list = pickle.load(file)
            return my_list
        except Exception as e:
            print(f"An error occurred while reading the list: {e}")
            return None