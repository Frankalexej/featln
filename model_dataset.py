import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

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
        return inp, outp