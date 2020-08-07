import torch
import numpy as np

import os
import time
import argparse
import yaml

import ised

from ised.ised import Preprocessor
import openl3

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import seaborn as sns

import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

def do_fischer_reweighting(features, labels):
    raise NotImplementedError

class OpenL3:
    def __init__(self, input_repr='mel128', embedding_size=512, content_type='music'):
        print(f'initing openl3 with rep {input_repr}, embedding {embedding_size}, content {content_type}')
        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            embedding_size=embedding_size,
            content_type=content_type
        )

    def __call__(self, x, sr):
        x = ised.utils.assert_numpy(x)
        x = x.squeeze(0)
        # remove channel dimensions
        embedding, ts = openl3.get_audio_embedding(x, sr, model=self.model,  verbose=False,
                                             content_type="music", embedding_size=512,
                                             center=True, hop_size=1)

        if embedding.ndim > 1:
            if embedding.shape[0] > 1:
                embedding = embedding.mean(axis=0)
            elif embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

        assert embedding.ndim == 1

        return embedding

class VGGish:
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def __call__(self, x, sr):
        x = ised.utils.assert_numpy(x)
        x = x.squeeze(0)
        # lets do the mean var, dmean dvar on the vggish embedding
        embedding = self.model(x, sr)

        if embedding.ndim > 1:
            if embedding.shape[0] > 1:
                embedding = embedding.mean(axis=0)
            elif embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

        assert embedding.ndim == 1

        return embedding

def load_preprocessor(name: str):
    if name == 'vggish':
        model = VGGish()
    elif name == 'ised_features': # bongjun's ised model
        model = Preprocessor(
            sr=8000,
            mfcc_kwargs=dict(
                log_mels=False, 
                n_mfcc=13
            ),
            normalize=False
        )
    elif 'openl3' in name:
        params = name.split('-')
        model = OpenL3(
            input_repr=params[1],
            embedding_size=params[2], 
            content_type=params[3], 
        )
    else:
        raise ValueError("couldn't find preprocessor name")
    return model

def zero_pad(audio, length):
    """
    make sure audio is at least 1 second long
    """
    if audio.size()[1] < length:
        l = audio.size()[1]
        z = torch.zeros(length - l)
        audio = torch.cat([audio[0], z])
        audio = audio.unsqueeze(0)

    return audio

def dim_reduce(emb, labels, save_path, n_components=3, method='umap', title_prefix = ''):
    """
    dimensionality reduction for visualization!
    parameters:
        emb (np.ndarray): the samples to be reduces with shape (samples, features)
        labels (list): list of labels for embedding
        save_path (str): root directory of where u wanna save ur figure
        method (str): umap, tsne, or pca
        title_prefix (str): title for your figure

    """
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError

    proj = reducer.fit_transform(emb)

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            instrument=labels
        ))
        fig = px.scatter(df, x='x', y='y', color='instrument',
                        title=title_prefix+f"_{method}")

    elif n_components ==3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            instrument=labels
        ))
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='instrument',
                        title=title_prefix+f"_{method}")
    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.write_html(save_path + '/' + title_prefix+f"_{method}.html")

def downmix(audio):
    # downmix if neededa
        if a.ndim == 2:
            a = a.mean(axis=0)
    
def main(params):
    # timing
    tic = time.time()

    # --------------------------------------------
    # SETUP
    # --------------------------------------------
    # extract our params
    classes = tuple(params['classes'])

    # now, load our preprocessor
    preprocessor = load_preprocessor(params['preprocessor'])

    # load our training and test dataset
    path_to_csv = './data/philharmonia/all-samples/metadata.csv'
    dataset = ised.datasets.PhilharmoniaSet(path_to_csv, classes)
    train_loader, val_loader = ised.datasets.train_test_split(
        dataset, batch_size=1, val_split=0.3, shuffle=True,
        random_seed=params['seed'])


    if params['max_train'] is None:
        params['max_train'] = len(train_loader)

    # --------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------
    print(f'train set is {params["max_train"]} samples')

    # now, train our ised model
    train_features = []
    train_labels  = []
    for idx, sample in enumerate(train_loader):
        if idx > params['max_train']:
            break
        # remove batch dimension
        sample = ised.datasets.debatch(sample)
        audio = sample['audio']
        sr = sample['sr']
        label = sample['label']

        # prepare audio
        audio = zero_pad(audio)
        audio = downmix(audio)

        feature_vector = preprocessor(audio, sr)
        train_features.append(feature_vector)
        labels.append(label)

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['fischer_reweighting']:
        train_features = do_fischer_reweighting(train_features, train_labels) 

    # --------------------------------------------
    # CLASSIFIER SETUP AND PCA ON MODEL
    # --------------------------------------------
    # we will need one classifier per label

    pca = PCA(n_components=params['n_components'])
    X = pca.fit_transform(train_features, train_labels)

    classifier = KNeighborsClassifier(
        n_neighbors=params['n_neighbors'],  
    )

    # fit our classifier
    classifier.predict(X, labels)

    # now, train our ised model
    test_features = []
    test_labels  = []
    for idx, sample in enumerate(val_loader)):
        if idx > int(0.7 * params['max_train']):
            break
        # remove batch dimension
        sample = ised.datasets.debatch(sample)
        audio = sample['audio']
        sr = sample['sr']
        label = sample['label']

        # prepare audio
        audio = zero_pad(audio)
        audio = downmix(audio)

        feature_vector = preprocessor(audio, sr)
        
        test_features.append(feature_vector)
        test_labels.append(label)

    # dim reduce our test set and predict
    test_X = pca.transform(test_features)

    # make our predictions
    test_predict = classifier.predict(test_X)
    # --------------------------------------------
    # VALIDATION
    # --------------------------------------------
    # now, measure precision/accuracy with the validation set
    metrics = {}

    yt = test_labels
    yp = test_predict

    m = sklearn.metrics.confusion_matrix(yt, yp)
    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(yt, yp)
    metrics['precision'] = sklearn.metrics.precision_score(yt, yp)
    metrics['recall'] = sklearn.metrics.recall_score(yt, yp)
    metrics['f1'] = sklearn.metrics.f1_score(yt, yp)

    params['metrics'] = metrics

    print('experiment done!')

    output = dict(
        name=params['name'],
        seed=params['seed'],
        preprocessor=params['preprocessor']
        fischer_reweighting=params['fischer_reweighting']
        n_components=params['n_components']
        n_neighbors=params['n_neighbors']
        metrics=params['metrics']
    )

    output = ised.utils.flatten_dict(output)
    df = pd.DataFrame([output])
    df.to_csv(os.path.join(params['output_path'], 'output.csv'), index=False)

    # timing
    toc = time.time()
    print(f'experiment took {toc - tic} s\n')
    return output


def run(path_to_trials):
    for root, dirs, files in os.walk(path_to_trials, topdown=False):
        # every iteration of this on a specific depth
        # since its topdown, the first depth will be single run level
        output = []
        dirs.sort()
        for d in dirs:
            out_path = os.path.join(root, d, 'output.csv')
            if os.path.exists(out_path):
                df = pd.read_csv(out_path).to_dict('records')
                output.extend(df)
                o = pd.DataFrame(output)
                o.to_csv(os.path.join(root, 'output.csv'), index=False)
        for file in files:
            file = os.path.join(root, file)
            if file[-5:] == '.yaml':
                if os.path.exists(os.path.join(root, 'output.csv')):
                    print(f'already found output for {file}. passing')
                    continue
                with open(file, 'r') as f:
                    params = yaml.load(f)
                    print(f'running exp with name {params["name"]}')
                    params['output_path'] = os.path.join(root)
                    main(params)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_experiment', '-p',
                        default='week3/experiments')

    args = parser.parse_args()
    exp_path = args.path_to_experiment

    print(exp_path)

    run(exp_path)