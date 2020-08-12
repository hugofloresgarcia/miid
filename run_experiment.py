import torchaudio
import torch
import numpy as np

import os
import time
import argparse
import yaml
from natsort import natsorted, ns

from philharmonia_dataset import PhilharmoniaSet, train_test_split, debatch

import labeler
from labeler.preprocessors import ISED_Preprocessor, OpenL3, VGGish

import pandas as pd
import sklearn.metrics

import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def do_fischer_reweighting(features, labels):

    p = [feature for feature, label in zip(features, labels) if label == 1]
    n = [feature for feature, label in zip(features, labels) if label == 0]
    p = np.array(p)
    n = np.array(n)

    weights = (p.mean(axis=0) ** 2 - n.mean(axis=0) ** 2) / \
                                  (p.std(axis=0) ** 2 + n.std(axis=0) ** 2)

    # VGGISH FIX
    # TODO: the last element in the vggish embedding is always 255. (why)
    #   this breaks fischer's criterion because the std deviation
    #   will always be 0, so you end up dividing by 0
    #   I'm currently replacing the nan by 0 (meaning that the feature will
    #   have no weight at all). should I be doing this?
    for i, w in enumerate(weights): 
        if np.isnan(w):
            weights[i] = 0

    return weights

def load_classifier(name: str):
    if 'svm' in name:
        params = name.split('-')
        kernel = params[1]
        degree = params[2] if 'poly' in params else 3

        classifier = SVC(
            kernel=kernel,
            degree=int(degree)
        )
        return classifier
    elif 'knn' in name:
        params = name.split('-')
        classifier = KNeighborsClassifier(
            n_neighbors=int(params[1])
        )
        return classifier
    else:
        raise ValueError(f"couldn't find classifier name: {name}")

openl3_models = {}
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
        if name in openl3_models:
            return openl3_models[name]
        model = OpenL3(
            input_repr=params[1],
            embedding_size=int(params[2]), 
            content_type=params[3], 
        )
        openl3_models[name] = model
    else:
        raise ValueError("couldn't find preprocessor name")
    return model

def zero_pad(audio, length):
    """
    make sure audio is at least 1 second long
    """
    if audio.shape[0] < length:
        l = audio.shape[0]
        z = np.zeros(length - l)
        audio = np.concatenate([audio, z])

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

    elif n_components == 3:
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
    # downmix if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio
    
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
    path_to_dataset = './data/philharmonia/'
    dataset = PhilharmoniaSet(path_to_dataset, classes)
    train_loader, val_loader = train_test_split(
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
        sample = debatch(sample)
        label = sample['label'].item()
        path_to_audio = sample['path_to_audio']
        filename = sample['filename']

        # dynamically load embeddings!! (check if we already have it)
        embedding_root = os.path.join('embeddings', params['preprocessor'])
        embedding_name = filename[0:-4] + '.json'

        path_to_embedding = os.path.join(embedding_root, embedding_name)
        if os.path.exists(path_to_embedding):
            feature_vector = pd.read_json(path_to_embedding, typ='series').tolist()
            feature_vector = np.array(feature_vector)

            train_features.append(feature_vector)
            train_labels.append(label)
            continue
        
        # else, let's go ahead and load the audio
        audio, sr = torchaudio.load(path_to_audio)
        audio = audio.detach().numpy()

        # prepare audio
        audio = downmix(audio)
        audio = zero_pad(audio, sr * 1) # we need the audio to be at least 1s

        feature_vector = preprocessor(audio, sr)

        # save our feature vector
        if not os.path.exists(embedding_root):
            os.mkdir(embedding_root)
        pd.Series(feature_vector).to_json(path_to_embedding)

        train_features.append(feature_vector)
        train_labels.append(label)

    train_features = np.stack(train_features, axis=0)
    train_labels = np.stack(train_labels, axis=0)

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['fischer_reweighting']:
        fischer_weights = do_fischer_reweighting(train_features, train_labels) 
        train_features = np.array([fischer_weights * f for f in train_features])

    # --------------------------------------------
    # CLASSIFIER SETUP AND PCA ON DATA
    # --------------------------------------------

    pca = PCA(n_components=params['pca_n_components'])
    #TODO: standardize features prior to pca
    X = pca.fit_transform(train_features, train_labels)
    # X = train_features

    classifier = load_classifier(params['classifier'])

    # fit our classifier
    classifier.fit(X, train_labels)

    test_features = []
    test_labels  = []
    for idx, sample in enumerate(val_loader):
        if idx > int(0.7 * params['max_train']):
            break
        # remove batch dimension
        sample = debatch(sample)
        label = sample['label'].item()
        path_to_audio = sample['path_to_audio']
        filename = sample['filename']

        # dynamically load embeddings!! (check if we already have it)
        embedding_root = os.path.join('embeddings', params['preprocessor'])
        embedding_name = filename[0:-4] + '.json'

        path_to_embedding = os.path.join(embedding_root, embedding_name)
        if os.path.exists(path_to_embedding):
            feature_vector = pd.read_json(path_to_embedding, typ='series').tolist()
            feature_vector = np.array(feature_vector)

            test_features.append(feature_vector)
            test_labels.append(label)
            continue
        
        # else, let's go ahead and load the audio
        audio, sr = torchaudio.load(path_to_audio)
        audio = audio.detach().numpy()

        # prepare audio
        audio = downmix(audio)
        audio = zero_pad(audio, sr * 1) # we need the audio to be at least 1s

        feature_vector = preprocessor(audio, sr)

        # save our feature vector
        if not os.path.exists(embedding_root):
            os.mkdir(embedding_root)
        pd.Series(feature_vector).to_json(path_to_embedding)

        test_features.append(feature_vector)
        test_labels.append(label)

    test_features = np.stack(test_features, axis=0)
    test_labels = np.stack(test_labels, axis=0)

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['fischer_reweighting']:
        test_features = np.array([fischer_weights * f for f in test_features])

    # dim reduce our test set and predict
    #TODO: standardize features prior to pca
    test_X = pca.transform(test_features)
    # test_X = test_features

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
        preprocessor=params['preprocessor'],
        fischer_reweighting=params['fischer_reweighting'],
        pca_n_components=params['pca_n_components'],
        classifier=params['classifier'],
        metrics=params['metrics']
    )

    output = labeler.utils.flatten_dict(output)

    # timing
    toc = time.time()
    print(f'experiment took {toc - tic} s\n')

    return output

def run_trials(path_to_trials):

    for root, dirs, files in os.walk(path_to_trials, topdown=False):
        # traverse directory tree top down, collecting an output.csv at every level. 
        output = []

        out_path = os.path.join(root, 'output.csv')
        if os.path.exists(out_path):
            print(out_path)
            df = pd.read_csv(out_path).to_dict('records')
            output.extend(df)

        dirs = natsorted(dirs, alg=ns.IGNORECASE)
        for d in dirs:
            out_path = os.path.join(root, d, 'output.csv')
            if os.path.exists(out_path):
                df = pd.read_csv(out_path).to_dict('records')
                output.extend(df)
                o = pd.DataFrame(output)
                o.to_csv(os.path.join(root, 'output.csv'), index=False)
                del o 

        # output = []
        files = natsorted(files, alg=ns.IGNORECASE)
        for file in files:
            file = os.path.join(root, file)
            
            if file[-5:] == '.yaml':
                # if os.path.exists(os.path.join(root, 'output.csv')):
                #     print(f'already found output for {file}. passing')
                #     continue
                with open(file, 'r') as f:
                    params = yaml.load(f)
                    print(f'running exp with name {params["name"]}')

                    if params['name'] in [d['name'] for d in output]:
                        print(f'skipping {params["name"]} as it has already been found in output.csv\n')
                        continue

                    params['output_path'] = os.path.join(root)
                    out  = main(params)
                    output.append(out)


        o = pd.DataFrame(output)
        o.to_csv(os.path.join(root, 'output.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_experiment', '-p',
                        default='week3/experiments')

    args = parser.parse_args()
    exp_path = args.path_to_experiment

    print(exp_path)

    run_trials(exp_path)