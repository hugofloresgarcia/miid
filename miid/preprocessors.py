from .core import compute_ISED_features

import openl3
import torch
import numpy as np

def _input_audio_check(x: np.ndarray):
    assert isinstance(x, np.ndarray), "input needs to be a numpy array"
    assert x.ndim == 2, "audio needs to be (channels, samples)"
    assert x.shape[-1] > x.shape[0], "audio needs to be (channels, samples)"

def load_preprocessor(name: str):
    if name == 'vggish':
        model = VGGish()
    elif name == 'ised': # bongjun's ised model
        model = ISED_Preprocessor(
            sr=8000,
            mfcc_kwargs=dict(
                log_mels=False, 
                n_mfcc=13
            ),
            normalize=False
        )
    elif 'openl3' in name:
        params = name.split('-')
        # this fixes all the memory leakage coming from loading an openl3 model
        if name in OpenL3.models:
            return OpenL3.models[name]

        model = OpenL3(
            input_repr=params[1],
            embedding_size=int(params[2]), 
            content_type=params[3], 
        )
        OpenL3.models[name] = model
    else:
        raise ValueError(f"couldn't find preprocessor name {name}")
    return model

class ISED_Preprocessor:
    def __init__(self, sr, mfcc_kwargs, normalize=False):
        self.sr = sr
        self.mfcc_kwargs = mfcc_kwargs
        self.normalize = normalize

    def __call__(self,  x: np.ndarray, sr: int):
        _input_audio_check(x)
        feats = compute_ISED_features(x, sr, self.sr, self.mfcc_kwargs, self.normalize)
        return feats

class OpenL3:
    models = {}
    
    def __init__(self, input_repr: str = 'mel128',
                       embedding_size: int = 512, 
                       content_type: str = 'music'):

        print(f'initing openl3 with rep {input_repr}, embedding {embedding_size}, content {content_type}')

        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            embedding_size=embedding_size,
            content_type=content_type
        )

    def __call__(self, x: np.ndarray, sr: int):
        _input_audio_check(x)

        x = x.mean(axis=0, keepdims=False) # downmix to mono
        embedding, ts = openl3.get_audio_embedding(x, sr, model=self.model)

        assert embedding.ndim == 2
        return embedding

class VGGish:
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def __call__(self, x: np.ndarray, sr: int):

        # lets do the mean var, dmean dvar on the vggish embedding
        embedding = self.model(x, sr)
        embedding = embedding.detach().numpy()

        assert embedding.ndim == 2
        return embedding