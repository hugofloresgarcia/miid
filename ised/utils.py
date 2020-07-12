import matplotlib.pyplot as plt
import numpy as np
import torch 
import torchaudio
import datetime


def assert_numpy(x):
    """
    make sure we're getting a numpy array (not a torch tensor)
    fix it if we need to
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    else:
        x = x

    assert isinstance(x, np.array)
    return x

def get_time():
    """
    return a date string w dir-compatible formatting (no spaces or slashes)
    """
    t = datetime.datetime.now()
    return ('_').join(t.strftime('%x').split('/')) + '_'+('.').join(t.strftime('%X').split(':'))


class ResampleDownmix:
    def __init__(self, old_sr: int, sr: int):
        # resample 
        self.resample = torchaudio.transforms.Resample(old_sr, sr)
        
    def __call__(self, audio: torch.Tensor):
        # downmix from stereo if needed
        if audio.size()[-2] == 2:
            audio = audio.mean(dim=(-2,))
            
        audio = self.resample(audio)
        return audio