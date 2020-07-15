import torch
import numpy as np

import yaml
import os
import argparse

import ised
from ised import plot_utils
from ised.ised import Preprocessor, Model

import pandas as pd
import sklearn.metrics


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
    else:
        raise NameError("couldn't find preprocessor name")
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

    return audio


def run_experiment(exp):

    # extract our params
    sr = exp['sr']
    # add our n_fft from window size param
    window_size = exp['window_size']
    if exp['preprocessor']['name'] == 'ised_features':
        exp['preprocessor']['sr'] = exp['sr']
        exp['preprocessor']['mfcc_kwargs']['melkwargs'] = {}
        exp['preprocessor']['mfcc_kwargs']['melkwargs']['n_fft'] = int(window_size * sr)
    # our classes
    classes = tuple(exp['classes'])

    # now, load our preprocessor
    preprocessor = load_preprocessor(exp['preprocessor'])

    # load our training and test dataset
    path_to_csv = './data/philharmonia/all-samples/metadata.csv'
    dataset = ised.datasets.PhilharmoniaSet(path_to_csv, classes)
    train_loader, val_loader = ised.datasets.train_test_split(
        dataset, batch_size=1, val_split=0.3, shuffle=True,
        random_seed=42)

    # load our initial target
    target = dataset.get_example(classes[0])

    # forward pass our target through the preprocessor
    # to get a feature vector
    target['features'] = preprocessor(target['audio'], target['sr'])

    # load our ised model with our target
    model = Model(target['features'], label=target['instrument'])

    # now, train our ised model
    for idx, sample in enumerate(train_loader):
        if exp['max_train'] is not None:
            if idx > exp['max_train']:
                break
        # remove batch dimension
        sample = ised.datasets.debatch(sample)

        # forward pass
        sample['audio'] = trim_or_pad(sample['audio'], sample['sr'])

        sample['features'] = preprocessor(sample['audio'], sample['sr'])

        # add example to our model
        model.add_example(sample['features'], sample['instrument'])

    model.reweigh_features()  # weigh our features

    # we will need one classifier per label
    classifiers = {}

    # now, let's log our PCA with weights/noweights
    for label in model.get_labels():
        # get a model pca
        pmap, nmap = model.do_pca(label, exp['pca']['num_components'], exp['model']['weights'])
        fig, axes = plot_utils.get_figure(1, 2,
                                          title=f'PCA_{label}_{exp["preprocessor"]["name"]}')

        # plot pca
        axes[0] = plot_utils.plot_pca(axes[0], {label: pmap, f'not_{label}': nmap})
        if exp['model']['weights']:
            W = model.weights[label]
        else:
            W = np.ones(model.weights[label].shape)
        # plot features
        axes[1] = plot_utils.plot_features(axes[1], W, title='fischer weights')

        # save fig
        fig.savefig(
            ised.utils.mkdir(f"{exp['output_dir']}/{exp['name']}") + '/' + f'{label}'
        )

        # add a classifier
        plabels = [label for e in pmap]
        nlabels = [f'not {label}' for e in nmap]
        labels = np.array(plabels + nlabels)

        data = np.array(list(pmap) + list(nmap))
        # add a KNN classifier for this particular label
        classifiers[label] = ised.neighbors.KNN(data, labels)

    # now, measure precision/accuracy with the validation set
    metrics = {}
    yt = []
    yp = []
    print(f'validating with {len(val_loader)} samples')
    for idx, sample in enumerate(val_loader):
        # remove batch dimension
        sample = ised.datasets.debatch(sample)

        # forward pass
        sample['audio'] = trim_or_pad(sample['audio'], sample['sr'])
        sample['features'] = preprocessor(sample['audio'], sample['sr'])

        # now, do classification by label
        sample['scores'] = {}
        y_pred = []
        y_true = []
        for label in model.get_labels():
            # do pca on weighed features
            x = model(sample['features'], label)
            x = ised.utils.assert_numpy(x)
            x = model.pca[label].transform(x.reshape(1, -1))

            # predict using KNN
            pred = classifiers[label].predict(x, exp['num_neighbors'])
            target = label if label == sample['instrument'] else f'not {label}'

            y_pred.append(1 if pred == label else 0)
            y_true.append(1 if target == label else 0)
            # print(f'GROUND TRUTH: {sample["instrument"]}\tPRED: {pred}\tTARGET: {target}')
        yt.append(y_true)
        yp.append(y_pred)

    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(yt, yp)
    metrics['precision'] = sklearn.metrics.precision_score(yt, yp, average='micro')
    metrics['recall'] = sklearn.metrics.recall_score(yt, yp, average='micro')
    metrics['f1'] = sklearn.metrics.f1_score(yt, yp, average='micro')

    exp['metrics'] = metrics
    print('experiment done!')

    return exp


def load_experiments(exp_path, exp_list):
    if os.path.isdir(exp_path):
        for root, dirs, files in os.walk(exp_path):
            for name in files:
                if name[-5:] == '.yaml':
                    filepath = os.path.join(root, name)
                    with open(filepath, 'r') as f:
                        exp = yaml.load(f)
                        exp_list.append(exp)
            for dir in dirs:
                new_root = os.path.join(root, dir)
                load_experiments(new_root, exp_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_experiments',
                        default='week3/experiments')

    args = parser.parse_args()
    exp_path = args.path_to_experiments

    # with open('week3/config.yaml', 'r') as f:
    #     experiments = yaml.load(f)

    # LOAD EXPERIMENTS
    experiments = []
    outputs = []
    print('loading experiments...')
    load_experiments(exp_path, experiments)
    print(f'found {len(experiments)} experiments')
    for exp in experiments:
        ised.utils.pretty_print(exp)
        out = run_experiment(exp)
        out = ised.utils.flatten_dict(out)
        outputs.append(out)
        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(exp_path, 'output.csv'))


