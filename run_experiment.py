import torch
import numpy as np
import matplotlib.pyplot as plt

import openl3

import yaml
import os
import argparse

import ised
from ised import plot_utils
from ised.ised import Preprocessor, Model

import pandas as pd
import sklearn.metrics
import seaborn as sns

import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ised.utils import ResampleDownmix

import time

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

def load_preprocessor(params: dict):
    name = params['name']

    if name == 'vggish':
        model = VGGish()
    elif name == 'ised_features':
        model = Preprocessor(
            sr=params['sr'],
            mfcc_kwargs=params['mfcc_kwargs'],
            normalize=params['normalize']
        )
    elif name == 'openl3':
        model = OpenL3(**params['openl3_kwargs'])
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

def main(params):
    # timing
    tic = time.time()

    # --------------------------------------------
    # SETUP
    # --------------------------------------------
    # extract our params
    sr = params['sr']
    # add our n_fft from window size param
    window_size = params['window_size']
    if params['preprocessor']['name'] == 'ised_features':
        params['preprocessor']['sr'] = params['sr']
        params['preprocessor']['mfcc_kwargs']['melkwargs'] = {}
        params['preprocessor']['mfcc_kwargs']['melkwargs']['n_fft'] = int(window_size * sr)
    # our classes
    classes = tuple(params['classes'])

    # now, load our preprocessor
    preprocessor = load_preprocessor(params['preprocessor'])

    # load our training and test dataset
    path_to_csv = './data/philharmonia/all-samples/metadata.csv'
    dataset = ised.datasets.PhilharmoniaSet(path_to_csv, classes)
    train_loader, val_loader = ised.datasets.train_test_split(
        dataset, batch_size=1, val_split=0.3, shuffle=True,
        random_seed=params['seed'])

    # load our initial target
    target = dataset.get_example(classes[0])

    # forward pass our target through the preprocessor
    target['audio'] = zero_pad(target['audio'], target['sr'])
    target['audio'] = ResampleDownmix(target['sr'], target['sr'])(target['audio'])
    target['features'] = preprocessor(target['audio'], target['sr'])

    # load our ised model with our target
    model = Model(target['features'], label=target['instrument'])

    if params['max_train'] is None:
        params['max_train'] = len(train_loader)

    # --------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------
    print(f'training on {params["max_train"]} samples')
    # now, train our ised model
    for idx, sample in enumerate(train_loader):
        if idx > params['max_train']:
            break
        # remove batch dimension
        sample = ised.datasets.debatch(sample)

        # forward pass
        sample['audio'] = zero_pad(sample['audio'], sample['sr'])
        sample['audio'] = ResampleDownmix(sample['sr'], sample['sr'])(sample['audio'])
        sample['features'] = preprocessor(sample['audio'], sample['sr'])

        # add example to our model
        model.add_example(sample['features'], sample['instrument'])

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['model']['weights']:
        model.reweigh_features()  # weigh our features

    # --------------------------------------------
    # KNN SETUP AND PCA ON MODEL
    # --------------------------------------------
    # we will need one classifier per label
    classifiers = {}

    # now, let's log our PCA with weights/noweights
    for label in model.get_labels():
        # get a model pca
        pmap, nmap = model.do_pca(label, params['num_components'], params['model']['weights'])

        # add a classifier
        plabels = [label for e in pmap]
        nlabels = [f'not {label}' for e in nmap]
        labels = np.array(plabels + nlabels)

        data = np.array(list(pmap) + list(nmap))
        # add a KNN classifier for this particular label
        classifiers[label] = ised.neighbors.KNN(data, labels)

    # --------------------------------------------
    # VALIDATION
    # --------------------------------------------
    # now, measure precision/accuracy with the validation set
    metrics = {}
    yt = []
    yp = []
    wrong = {}
    right = {}
    print(f'validating with {int(0.7 * params["max_train"])} samples')
    for idx, sample in enumerate(val_loader):
        if idx > int(0.7 * params['max_train']):
            break
        # remove batch dimension
        sample = ised.datasets.debatch(sample)

        # forward pass
        sample['audio'] = zero_pad(sample['audio'], sample['sr'])
        sample['audio'] = ResampleDownmix(sample['sr'], sample['sr'])(sample['audio'])
        sample['features'] = preprocessor(sample['audio'], sample['sr'])

        # now, do classification by label
        y_pred = []
        y_true = []
        for label in model.get_labels():
            # do pca on weighed features
            x = model(sample['features'], label)
            x = ised.utils.assert_numpy(x)
            x = model.pca[label].transform(x.reshape(1, -1))

            if label not in wrong:
                wrong[label] = []
            if label not in right:
                right[label] = []

            # predict using KNN
            pred = classifiers[label].predict(x, params['num_neighbors'])
            target = label if label == sample['instrument'] else f'not {label}'

            if pred == target:
                right[label].append(x.squeeze(0).tolist())
            else:
                wrong[label].append(x.squeeze(0).tolist())

            y_pred.append(1 if pred == label else 0)
            y_true.append(1 if target == label else 0)

        yt.append(y_true)
        yp.append(y_pred)

    # --------------------------------------------
    # ROUND OF DIMENSION REDUCTION
    # --------------------------------------------
    for label in model.get_labels():
        X, labels = model.get_features_with_labels(label, weights=params['model']['weights'])
        methods = ['pca']

        fig, ax = plot_utils.get_figure(1, 1, title=f'{label}_weights')
        W = model.weights[label]
        #     # plot features
        ax = plot_utils.plot_features(ax, W, title='fischer weights')
        fig.savefig(ised.utils.mkdir(f"{params['output_dir']}") + '/' + f'{label}_weights')

        for method in methods:
            dim_reduce(X, labels, ised.utils.mkdir(f"{params['output_dir']}"),
                       method=method, n_components=params['num_components'],
                       title_prefix=f'{params["preprocessor"]["name"]}_{label}')


    # AHHH I CAN'T BELIEVE I DID THIS WRONG THE FIRST TIME
    yt = np.argmax(yt, axis=1)
    yp = np.argmax(yp, axis=1)

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
        weights=params['model']['weights'],
        preprocessor=params['preprocessor']['name'],
        num_classes=len(dataset.classes),
        metrics=metrics,
        pca_components=params['num_components'],
        neighbors=params['num_neighbors']
    )
    if params['preprocessor']['name'] == 'openl3':
        output['preprocessor'] += '_' + params['preprocessor']['openl3_kwargs']['input_repr']
        output['preprocessor'] += '_' + params['preprocessor']['openl3_kwargs']['content_type']
        output['preprocessor'] += '_' + str(params['preprocessor']['openl3_kwargs']['embedding_size'])

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