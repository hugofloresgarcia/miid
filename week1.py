import torch
import torchaudio
from ised import core as ised
import ised.utils as utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import argparse

def main(path_to_audio, params):
    assert os.path.exists(path_to_audio), "couldn't find that path :("
    
    sr = params['sr']
    chunk_len = params['chunk_len']
    mfcc_kwargs = params['mfcc_kwargs']
    normalize_features = params['normalize_features']
    
    # load audio
    filepath = path_to_audio
    print('loading audio...')
    audio, old_sr = torchaudio.load(filepath)
    print('audio loaded :)')


    # downmix from stereo if needed
    if audio.size()[-2] == 2:
        print('found stereo mix. downmixing')
        audio = audio.mean(dim=(-2,))
        # restore channel dimension after downmixing
        audio = audio.view(1, audio.size()[-1])

    # resample 
    torchaudio.transforms.Resample(old_sr, sr)(audio)

    # reshape our audio into 1 second chunks
    chunk_len = int(chunk_len * sr)
    end = len(audio[0]) // chunk_len * chunk_len
    audio = audio[:, 0:end]
    audio = audio.view((-1, chunk_len))

    # do mfcc
    MFCC = torchaudio.transforms.MFCC(**mfcc_kwargs)

    # create our feature matrix
    features = torch.Tensor([ised.features(MFCC(chunk)).tolist() for chunk in audio])

    # normalize
    if normalize_features:
        features = (features - features.mean())/features.std()

    
    sk_pca = PCA(2)
    sk_features_redux = sk_pca.fit(features).transform(features)

    # plot
    fig = plt.figure()
    axes = fig.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    axes[0].scatter(sk_features_redux[:, 0], sk_features_redux[:, 1])
    axes[0].set_title('PCA')

    axes[1].imshow(sk_pca.get_covariance())
    axes[1].set_title('covariance')
    fig.savefig(params['output_path'])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_audio',
                       help='path to audio file to analyze')
    
    parser.add_argument('--sample_rate', '-s',
                        type=int, default=8000,
                        help='sample rate used for analysis (default: 8000)')
    
    parser.add_argument('--chunk_len', '-l',
                       type=float, default=1, 
                       help='chunk length of analysis, in seconds (default: 1)')
    
    parser.add_argument('--n_mfcc',
                       type=int, default=13, 
                       help='number of MFCCs (default: 13)')
    
    parser.add_argument('--db_mels',
                       action='store_true', 
                       help='use dB scaled mel spectrogram instead of log')
    
    parser.add_argument('--n_fft', 
                       type=float, default=90e-3, 
                       help='fft window window size (in s) (default: 90e-3)')
    
    parser.add_argument('--no_normalization', '-n',
                        action='store_true', 
                       help='skip normalization of feature vectors to mean 0 and std 1')
    
    parser.add_argument('--output_path', '-o', 
                        type=str, default='week1_output.jpg',
                       help='path to save plots to')
    
    args = parser.parse_args()
    
    params = {}
    
    params['sr'] = args.sample_rate
    params['chunk_len'] = args.chunk_len
    
    params['mfcc_kwargs'] = {}
    params['mfcc_kwargs']['n_mfcc'] =  args.n_mfcc
    params['mfcc_kwargs']['log_mels'] = not args.db_mels
    
    params['mfcc_kwargs']['melkwargs'] = {}
    params['mfcc_kwargs']['melkwargs']['n_fft'] = \
                        int(args.n_fft * args.sample_rate)
    params['normalize_features'] =  not args.no_normalization
    params['output_path'] = args.output_path
    
    print(f'PARAMS:\n{params}')
    
    main(args.path_to_audio, params)
    print('done!')
    print(f'output written to {params["output_path"]}')