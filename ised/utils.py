import numpy as np
import torch 
import torchaudio
import datetime
import os
import yaml
import collections

def flatten_dict(d, parent_key='', sep='_'):
    """
    took this from
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mkdir(path):
    if os.path.exists(path):
        return path

    os.makedirs(path)
    return path


def pretty_print(dictionary):
    data = yaml.dump(dictionary, default_flow_style=False)
    print(data)

def assert_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError("input was neither an np array or tensor")

def assert_numpy(x):
    """
    make sure we're getting a numpy array (not a torch tensor)
    fix it if we need to
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    else:
        x = x

    assert isinstance(x, np.ndarray)
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