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

class OpenL3:
    def __init__(self):
        self.model = openl3.models.load_audio_embedding_model(
            input_repr='mel128',
            embedding_size=512,
            content_type='music'
        )

    def __call__(self, x, sr):
        x = trim_or_pad(x, sr)
        x = ised.utils.assert_numpy(x)
        # remove channel dimensions
        emb, ts = openl3.get_audio_embedding(x, sr, model=self.model,  verbose=False,
                                             content_type="music", embedding_size=512,
                                             center=True, hop_size=0.1
                                             )
        # print(f"EMB DIM: {emb.shape}")
        return emb.squeeze(0)

class VGGish:
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def __call__(self, x, sr):
        x = trim_or_pad(x, sr)
        # remove channel dimension and convert to numpy if needed
        x = x.squeeze(0)
        x = ised.utils.assert_numpy(x)
        # lets do the mean var, dmean dvar on the vggish embedding
        v = self.model(x, sr)

        # TODO: this is hacky. I shouldn't have to do this
        if not v.shape[0] == 128:
            v = v[0]

        return v


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
        model = OpenL3()
    else:
        raise ValueError("couldn't find preprocessor name")
    return model


def trim_or_pad(audio, length):
    # TODO: do I want to trim the audio at 1s?
    if audio.size()[1] < length:
        l = audio.size()[1]
        z = torch.zeros(length - l)
        audio = torch.cat([audio[0], z])
        audio = audio.unsqueeze(0)
    elif audio.size()[1] > length:
        audio = audio[:, 0:length]
        # if audio.size()[1] > 1.5 * length:
        #     audio = audio[:, int(0.5*length):int(1.5*length)]
        # else:
        #     audio = audio[:, 0:length]

    return audio


def main(params):
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
    # to get a feature vector
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
        sample['audio'] = trim_or_pad(sample['audio'], sample['sr'])
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
        pmap, nmap = model.do_pca(label, params['pca']['num_components'], params['model']['weights'])

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
        sample['audio'] = trim_or_pad(sample['audio'], sample['sr'])
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
    # ROUND OF PCA PLOTS
    # --------------------------------------------
    for label in model.get_labels():
        # get a model pca
        pmap, nmap = model.do_pca(label, params['pca']['num_components'], params['model']['weights'])
        fig, axes = plot_utils.get_figure(2, 2,
                                          title=f'{params["name"]}_{label}_{params["preprocessor"]["name"]}')

        r = np.array(right[label])
        w = np.array(wrong[label])

        # # plot pca
        # axes[0][0] = plot_utils.plot_pca(axes[0][0], {label: pmap, f'not_{label}': nmap,
        #                                               'right': r, 'wrong': w}, title='train and test')

        axes[1][0] = plot_utils.plot_pca(axes[1][0], {label: pmap, f'not_{label}': nmap}, title='train set')
        axes[1][1] = plot_utils.plot_pca(axes[1][1], {'right': r, 'wrong': w}, title='test',
                                         colors=dict(right='g', wrong='r'))
        W = model.weights[label]
        # plot features
        axes[0][1] = plot_utils.plot_features(axes[0][1], W, title='fischer weights')

        # save fig
        fig.savefig(
            ised.utils.mkdir(f"{params['output_dir']}") + '/' + f'{label}_validation'
        )

        # close fig when we're done
        plt.close(fig)

        X_tsne, y = model.do_tsne(label, params['pca']['num_components'], params['model']['weights'])

        tsne_df = pd.DataFrame(data={
            'tsne_one': X_tsne[:, 0],
            'tsne_two': X_tsne[:, 1],
            'y': y
        })

        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.subplots(1)
        sns.scatterplot(
            x="tsne_one", y="tsne_two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=tsne_df,
            legend="full",
            alpha=0.8,
            ax=ax1
        )

        # save fig
        fig.savefig(
            ised.utils.mkdir(f"{params['output_dir']}") + '/' + f't-sne_{label}_validation'
        )
        plt.close(fig)

        X_umap, y = model.do_umap(label, params['pca']['num_components'], params['model']['weights'])

        umap_df = pd.DataFrame(data={
            'umap_one': X_umap[:, 0],
            'umap_two': X_umap[:, 1],
            'y': y
        })

        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.subplots(1)
        sns.scatterplot(
            x="umap_one", y="umap_two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=umap_df,
            legend="full",
            alpha=0.8,
            ax=ax1
        )

        # save fig
        fig.savefig(
            ised.utils.mkdir(f"{params['output_dir']}") + '/' + f'umap_{label}_validation'
        )
        plt.close(fig)

    # AHHH I CAN'T BELIEVE I DID THIS WRONG THE FIRST TIME
    yt = np.argmax(yt, axis=1)
    yp = np.argmax(yp, axis=1)

    m = sklearn.metrics.confusion_matrix(yt, yp)
    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(yt, yp)
    metrics['precision'] = sklearn.metrics.precision_score(yt, yp)
    metrics['recall'] = sklearn.metrics.recall_score(yt, yp)
    metrics['f1'] = sklearn.metrics.f1_score(yt, yp)

    params['metrics'] = metrics

    print('experiment done!\n')

    output = dict(
        name=params['name'],
        seed=params['seed'],
        weights=params['model']['weights'],
        preprocessor=params['preprocessor']['name'],
        num_classes=len(dataset.classes),
        metrics=metrics,
        pca_components=params['pca']['num_components'],
        neighbors=params['num_neighbors']
    )
    output = ised.utils.flatten_dict(output)
    df = pd.DataFrame([output])
    df.to_csv(os.path.join(params['output_path'], 'output.csv'), index=False)
    return output


def run(path_to_trials):

    for root, dirs, files in os.walk(path_to_trials, topdown=False):
        # every iteration of this on a specific depth
        # since its topdown, the first depth will be single run level
        output = []
        for dir in dirs:
            out_path = os.path.join(root, dir, 'output.csv')
            if os.path.exists(out_path):
                df = pd.read_csv(out_path).to_dict('records')
                output.extend(df)
                o = pd.DataFrame(output)
                o.to_csv(os.path.join(root, 'output.csv'), index=False)
        for file in files:
            file = os.path.join(root, file)
            if file[-5:] == '.yaml':
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