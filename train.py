import torchaudio
import torch
import numpy as np

import os
import time
import yaml
from natsort import natsorted, ns

from joblib import dump

import labeler
from labeler import audio_utils
from labeler.utils import save_confusion_matrix, dim_reduce
from labeler.preprocessors import load_preprocessor
from labeler.core import get_fischer_weights
from philharmonia_dataset import PhilharmoniaSet, train_test_split, debatch

import json
import pandas as pd
import sklearn.metrics

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.figure_factory as ff

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

def compute_embeddings(dataloader, preprocessor, embedding_root):
    """ compute preprocessor features and get labels from dataloader

    params:
        dataloader (torch DataLoader): dataloader to retrieve data from
        preprocessor (preprocessor object)
        embedding_root (root dir to look for precomputed embeddings)
    
    returns:

    """
    features = []
    labels = []
    label_indices = {}
    for idx, sample in enumerate(dataloader):

        print(f'processing {idx} out of {len(dataloader)}')

        # remove batch dimension
        sample = debatch(sample)
        label = sample['label'].item()
        path_to_audio = sample['path_to_audio']
        filename = sample['filename']

        label_indices[label] = sample['instrument']

        embedding_name = filename[0:-4] + '.json'

        path_to_embedding = os.path.join(embedding_root, embedding_name)
        if os.path.exists(path_to_embedding):
            print('using preloaded embedding\n')

            feature_vector = json.load(open(path_to_embedding))
            feature_vector = np.array([[v for k, v in val.items()] for key, val in feature_vector.items()]).T

            assert feature_vector.ndim == 2

            features.extend(feature_vector)
            # create an array of labels of the same shape as our feature vector
            # because we need a label every 1 second
            labels.extend(
                np.full_like(feature_vector[:, 0], label)
            )
            continue
        
        # else, let's go ahead and load the audio
        audio, sr = torchaudio.load(path_to_audio)
        audio = audio.detach().numpy()

        # prepare audio
        audio = audio_utils.downmix(audio)
        audio = audio_utils.zero_pad(audio, sr * 1) # we need the audio to be at least 1s

        # split out audio on silence
        audio_list, intervals = audio_utils.split_on_silence(audio)

        # this takes every split audio clip, preprocesses them and puts embeddings in an array shape (Frame, Feature)
        feature_vector = [preprocessor(aud, sr) for aud in audio_list]
        feature_vector = np.concatenate(feature_vector, axis=0)

        # save our feature vector
        if not os.path.exists(embedding_root):
            os.mkdir(embedding_root)
        pd.DataFrame(feature_vector).to_json(path_to_embedding)

        features.extend(feature_vector)
        labels.extend(
                np.full_like(feature_vector[:, 0], label)
        )

        print('\n')

    # convert to np array
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)

    return features, labels, [v for k,v in sorted(label_indices.items(), key=lambda x: x[0])]

def get_dataset_stats(labels, label_indices):
    """
    returns a dictionary with each class and class frequency
    params:
        labels (np.ndarray): np array of shape (samples,) with integer labels
        label_indices (list): lookup table with class names
    returns:
        dataset_stats (dict): dict with a count for each class name
    """
    dataset_stats = {label: 0 for label in label_indices}
    for label in labels:
        dataset_stats[label_indices[int(label)]] += 1
    return [dataset_stats]

# flags and hyperparams
classes = 'no_percussion'
preprocessor_name = 'openl3-mel256-6144-music'
classifier_name = 'svm-linear'
pca_n_components = 128
val_split = 0.3
random_seed = 42
output_path = 'results'
exp_name = 'exp'
fischer_reweighting = False

def train_sklearn_classifier(
        classes='no_percussion',
        preprocessor_name='openl3-mel256-6144-music',
        classifier_name='svm-linear',
        pca_n_components=128,
        val_split=0.3,
        random_seed=42,
        output_path='',
        exp_name='exp',
        fischer_reweighting=False):
    # timing
    tic = time.time()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # --------------------------------------------
    # SETUP
    # --------------------------------------------
    # extract our params
    if classes == 'no_percussion':
        classes = tuple("saxophone,flute,guitar,contrabassoon,bass-clarinet,trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))
    assert isinstance(classes, tuple)

    # now, load our preprocessor
    preprocessor = load_preprocessor(preprocessor_name)

    # load our training and test dataset
    path_to_dataset = './data/philharmonia/'
    dataset = PhilharmoniaSet(path_to_dataset, classes, load_audio=False)
    train_loader, val_loader = train_test_split(
                                        dataset, 
                                        batch_size=1, 
                                        val_split=val_split, 
                                        shuffle=True,
                                        random_seed=random_seed
                                        )

    print(dataset.get_class_data())

    # --------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------

    # now, train our ised model
    train_features, train_labels, label_indices = compute_embeddings(train_loader, 
                                    preprocessor=preprocessor,
                                    embedding_root=os.path.join('embeddings', 'silence_split', preprocessor_name))

    train_data_stats = get_dataset_stats(train_labels, label_indices)
    pd.DataFrame(train_data_stats).to_csv(
        os.path.join(output_path, 'train_dataset_stats.csv')
    )

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if fischer_reweighting:
        print('doing fischer reweighting...')

        fischer_weights = get_fischer_weights(train_features, train_labels) 
        train_features = np.array([fischer_weights * f for f in train_features])
        dump(fischer_weights, f'{output_path}/{exp_name}_fischer')

    # --------------------------------------------
    # CLASSIFIER SETUP AND PCA ON DATA
    # --------------------------------------------

    pca = PCA(n_components=pca_n_components)
    X = pca.fit_transform(train_features)
    dump(pca, f'{output_path}/{exp_name}_pca')

    classifier = load_classifier(classifier_name)

    # fit our classifier
    classifier.fit(X, train_labels)
    dump(classifier, f'{output_path}/{exp_name}_classifier')

    # --------------------------------------------
    # VALIDATION
    # --------------------------------------------

    test_features, test_labels, label_indices = compute_embeddings(
                                    dataloader=val_loader,
                                    preprocessor=preprocessor,
                                    embedding_root=os.path.join('embeddings', 'silence_split', preprocessor_name))

    test_data_stats = get_dataset_stats(test_labels, label_indices)
    pd.DataFrame(test_data_stats).to_csv(
        os.path.join(output_path, 'test_dataset_stats.csv')
    )
    
    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if fischer_reweighting:
        test_features = np.array([fischer_weights * f for f in test_features])

    # dim reduce our test set and predict
    test_X = pca.transform(test_features)

    # --------------------------------------------
    # PREDICT and METRICS
    # --------------------------------------------

    # make our predictions
    test_predict = classifier.predict(test_X)

    # now, measure precision/accuracy with the validation set
    metrics = {}

    yt = test_labels
    yp = test_predict

    m = sklearn.metrics.confusion_matrix(yt, yp)
    save_confusion_matrix(m, label_indices, 
                    save_path=os.path.join(output_path, 'confusion_matrix.html')
    )
    
    m_norm = sklearn.metrics.confusion_matrix(yt, yp, normalize='true')
    m_norm = np.around(m_norm, 3)
    save_confusion_matrix(m_norm, label_indices, 
                    save_path=os.path.join(output_path, 'confusion_matrix_normalized.html')
    )

    pd.DataFrame(m).to_csv(os.path.join(output_path, 'confusion_matrix.csv'))
    pd.DataFrame(label_indices).to_csv(os.path.join(output_path, 'label_indices.csv'))

    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(yt, yp)
    metrics['precision'] = sklearn.metrics.precision_score(yt, yp, average='macro')
    metrics['recall'] = sklearn.metrics.recall_score(yt, yp, average='macro')
    metrics['f1'] = sklearn.metrics.f1_score(yt, yp, average='macro')

    # logging for guild
    print(f'accuracy: {metrics["accuracy_score"]}')
    print(f'precision: {metrics["precision"]}')
    print(f'recall: {metrics["recall"]}')
    print(f'f1: {metrics["f1"]}')

    print('experiment done!')

    output = dict(
        exp_name=exp_name,
        seed=random_seed,
        preprocessor=preprocessor_name,
        fischer_reweighting=fischer_reweighting,
        pca_n_components=pca_n_components,
        classifier=classifier_name,
        **metrics
    )

    # timing
    toc = time.time()
    print(f'experiment took {toc - tic} s\n')

    pd.DataFrame([output]).to_csv(
        os.path.join(output_path, 'output.csv')
    )

    return output

if __name__ == "__main__":

    train_sklearn_classifier(
        classes=classes, 
        preprocessor_name=preprocessor_name, 
        classifier_name=classifier_name, 
        pca_n_components=pca_n_components, 
        val_split=val_split,
        random_seed=random_seed, 
        output_path=output_path, 
        exp_name=exp_name, 
        fischer_reweighting=fischer_reweighting)
