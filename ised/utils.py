import matplotlib.pyplot as plt
import numpy as np
import torch 
import torchaudio


def show_example(data: dict, jupyter=True):
    if 'instrument' in data:
        print(f"instrument: {data['instrument']}")
    if 'pitch' in data:
        print(f"pitch: {data['pitch']}")
    if 'features' in data:
        plot_features(data['features'])
    if 'audio' in data:
        plot_audio(data['audio'])
        

def plot_features(feature_vector: torch.Tensor, title='mfcc features',
                 output_dir=None, show=True):
    fv = feature_vector.detach().numpy()
    plt.title(title)
    plt.imshow(fv.reshape(-1, 4))
    plt.xticks([0, 1, 2, 3],
               ['mean', 'var', 'delta_mean', 'delta_var'],
               rotation=90)
    plt.colorbar()
    if output_dir is not None:
        plt.savefig(output_dir)
    if show:
        plt.show()


def plot_audio(audio: torch.Tensor, sr=None):
    a = audio.detach().view((-1,)).numpy()
    t = np.arange(len(a))
    
    xlabel = ''
    if sr:
        pass
    else:
        xlabel = 'samples'
        
    plt.plot(t, a)
    plt.xlabel(xlabel)
    plt.show()
    
    
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
    