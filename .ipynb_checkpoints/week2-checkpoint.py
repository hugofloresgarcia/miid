import torch
import torchaudio
from torch.utils.data import DataLoader
from ised import datasets
import ised.core as core
import ised.utils as utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.ioff()
import argparse


def compute_features(audio, old_sr):
    # downmix if needed and resample
    transform = utils.ResampleDownmix(old_sr, sr)(audio)
    # do mfcc
    mfcc = torchaudio.transforms.MFCC(**mfcc_kwargs)(audio)
    mfcc = mfcc.squeeze(0)

    # compute ised features for mfcc feature map
    features = core.features(mfcc)

    # normalize to mean 0 and std 1
    if normalize_features:
        features = (features - features.mean())/features.std()
    
    return features

def do_PCA(model, output_dir=None, weights=True):
    fig = plt.figure()
    
    if weights:
        axes = fig.subplots(1, 2)
        
        # get our complete feature map
        fmap = model.get_feature_map()
        # get negative and positive feature maps separately
        nmap = model.get_feature_map('negative')
        pmap = model.get_feature_map('positive')
        
        axes[1].imshow(model.weights.reshape(-1, 4))
        axes[1].set_title('feature weights')
    else:
        axes = fig.subplots(1, 2)
        # get our complete feature map
        fmap = model.get_subset(None, as_tensor=True)
        # get negative and positive feature maps separately
        nmap = model.get_subset('negative', as_tensor=True)
        pmap = model.get_subset('positive', as_tensor=True)
    
    # fit PCA to feature map
    pca = PCA(2)
    fmap = pca.fit(fmap).transform(fmap)

    # skip negative examples if we don't have any yet
    if not len(nmap) == 0:
        nmap = pca.transform(nmap)
        axes[0].scatter(nmap[:, 0], nmap[:, 1], 
                    label=other_classes)

    # show scatter
    pmap = pca.transform(pmap)
    axes[0].scatter(pmap[:, 0], pmap[:, 1], label=target_class)
    axes[0].legend()
    axes[0].set_title('PCA scatter')

    if output_dir is not None:
        fig.savefig(output_dir)
    
    # don't show
    plt.close(fig)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('classes', nargs='+', type=str, 
                       help='classes to obtain from dataset. run list_classes.py to see available classnames')
    parser.add_argument('--target', '-t', type=str, 
                        default=None,
                       help='target class. if none specified, first entry in classes will be used')
    parser.add_argument('--log_every', '-l', type=int, 
                        default=30,
                       help='save an image to output dir every... (default: 20)')
    parser.add_argument('--max_samples', '-m', type=int, 
                        default=300, 
                        help='maximum number of samples to obtain from dataset (default: 200)')
    parser.add_argument('--sample_rate', '-s',
                    type=int, default=8000,
                    help='sample rate used for analysis (default: 8000)')
    parser.add_argument('--n_mfcc',
                       type=int, default=13, 
                       help='number of MFCCs (default: 13)')
    
    args = parser.parse_args()
    if args.target is None:
        args.target = args.classes[0]
    
    
    # we only want to use 2 classes from the dataset
    dset_path = './data/philharmonia/all-samples/metadata.csv'
    classes = args.classes
    # params
    sr = args.sample_rate
    mfcc_kwargs = {
        'n_mfcc': args.n_mfcc,
        'log_mels': False,
        'melkwargs': {
            'n_fft': int(90e-3 * sr)
        }
    }
    normalize_features = False
    target_class = args.target
    log_every = args.log_every
    max_samples = args.max_samples

    other_classes = [c for c in classes if c != target_class]

    # load our dataset
    dset = datasets.PhilharmoniaSet(path_to_csv=dset_path, classes=classes)

    print(f'dataset with {len(dset.classes)} unique classes:')
    print(f'{dset.classes}')
    print(f'number of samples: {len(dset)}')

    dloader = DataLoader(dset, shuffle=True, batch_size=1)

    # retrieve our "target" example
    target = dset.get_example(target_class)
    # compute our feature vector (without ISED weights)
    target['features'] = compute_features(target['audio'], target['sr'])

#     utils.show_example(target, jupyter=False)

    # create an ISED model to keep and update our weights and relevances
    model = core.ISEDmodel(target=target['features'])

    # main training loop
    for i, sample in enumerate(dloader):
        if i > max_samples:
            break

        # debatch our samples
        sample = datasets.debatch(sample)
        # compute the features
        sample['features'] = compute_features(sample['audio'], sample['sr'])
        # check to see if our example will be positive or negative
        if sample['instrument'] == target['instrument']:
            model.add_example(sample['features'], 'positive')
        else:
            model.add_example(sample['features'], 'negative')

        fmap = model.get_feature_map()



        # show PCA every 
        if (i+1) % log_every == 0:
            # reweigh our feature vector
            model.reweigh_features()
            
            do_PCA(model, f'./week2_output/train_iter_{i+1}.jpg')

            print(f'recorded examples:\t{len(model.examples)-1}')
                
    # finally, compare no weights vs weight
    do_PCA(model, f'./week2_output/PCA_weights.jpg', weights=True)
    do_PCA(model, f'./week2_output/PCA_noweights.jpg', weights=False)
    
    exit(0)