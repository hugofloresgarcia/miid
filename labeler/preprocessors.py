from .core import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import openl3
import torch
import numpy as np

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

    def __call__(self, audio, sr):
        if isinstance(audio, np.ndarray):
            audio = torch.Tensor(audio)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        feats = compute_features(audio, sr, self.sr, self.mfcc_kwargs, self.normalize)

        return feats

class OpenL3:
    models = {}
    
    def __init__(self, input_repr='mel128', embedding_size=512, content_type='music'):
        print(f'initing openl3 with rep {input_repr}, embedding {embedding_size}, content {content_type}')
        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            embedding_size=embedding_size,
            content_type=content_type
        )

    def __call__(self, x, sr):
        assert isinstance(x, np.ndarray), "input needs to be a numpy array"
        assert isinstance(sr, int), "input needs to be an int"
        # remove channel dimensions
        embedding, ts = openl3.get_audio_embedding(x, sr, model=self.model,  verbose=False,
                                             content_type="music", embedding_size=512,
                                             center=False, hop_size=1)


        assert embedding.ndim == 2
        return embedding

class VGGish:
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def __call__(self, x, sr):
        assert isinstance(x, np.ndarray), "input needs to be a numpy array"
        assert isinstance(sr, int), "sr needs to be an int"
        # lets do the mean var, dmean dvar on the vggish embedding
        embedding = self.model(x, sr)
        embedding = embedding.detach().numpy()

        assert embedding.ndim == 2
        return embedding