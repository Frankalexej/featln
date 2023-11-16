import torch
import torch.nn as nn
import torchaudio

class MFCCTransform(nn.Module): 
    def __init__(self): 
        super().__init__()
    
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