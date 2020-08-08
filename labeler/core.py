import torch
import numpy as np
from . import utils
import torchaudio


def distance(a, b):
    """
    euclidean distance
    """
    return np.sqrt(np.sum((a - b) ** 2))

def delta(x: np.ndarray):
    """
    find delta values from MFCC
    """
    result = []
    for timestep in range(x.size()[1]-1):
        result.append((x[:, timestep] - x[:, timestep+1]).tolist())

    result = np.array(result).transpose(1, 0)
    return result

def features(mfcc: np.ndarray):
    """
    compute ISED feature vector from MFCCs
    """
    mfcc = utils.assert_torch(mfcc)
    dmfcc = delta(mfcc)
    
    fmean = mfcc.mean(axis=-1)
    fvar = mfcc.var(axis=-1)
    
    dmean = dmfcc.mean(axis=-1)
    dvar = dmfcc.var(axis=-1)
    
    return np.concatenate([fmean, fvar, dmean, dvar])

def relevance(s, sp, sn=None):
    """
    relevance score given positive and negative samples
    """
    if sn is None:
        return 1 / distance(s, sp)
    else:
        return distance(s, sn) / (distance(s, sn) + distance (s, sp))


def compute_features(audio, old_sr, sr, mfcc_kwargs, normalize=False):
    """
    get ISED feature vector from audio

    Parameters:
        audio: audio file with shape (channels, frames)
        old_sr: original audio sample rate
        mfcc_kwargs: keyworded arguments for torchaudio MFCC
    """
    # downmix if needed and resample
    utils.ResampleDownmix(old_sr, sr)(audio)
    # do mfcc
    mfcc = torchaudio.transforms.MFCC(**mfcc_kwargs)(audio)
    mfcc = mfcc.squeeze(0)

    # compute ised features for mfcc feature map
    feats = features(mfcc)

    # normalize to mean 0 and std 1
    if normalize:
        feats = (feats - feats.mean()) / feats.std()

    return utils.assert_numpy(feats)