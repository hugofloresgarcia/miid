import torchaudio
import torch
import numpy as np

import os
import time
import argparse
import yaml
from natsort import natsorted, ns

from joblib import dump

import labeler
from labeler.preprocessors import ISED_Preprocessor, OpenL3, VGGish
from philharmonia_dataset import PhilharmoniaSet, train_test_split, debatch

import json
import pandas as pd
import sklearn.metrics

import plotly.express as px
import plotly.figure_factory as ff

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_fischer_weights(features, labels):

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

def dim_reduce(emb, labels, save_path, n_components=3, method='umap', title=''):
    """
    dimensionality reduction for visualization!
    saves an html plotly figure to save_path
    parameters:
        emb (np.ndarray): the samples to be reduces with shape (samples, features)
        labels (list): list of labels for embedding
        save_path (str): path where u wanna save ur figure
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:    
        proj (np.ndarray): projection vector with shape (samples, dimensions)
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
                        title=title)
    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.write_html(save_path)
    return proj

def downmix(audio):
    # downmix if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio

def get_features_and_labels(dataloader, preprocessor, embedding_root):
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
        audio = downmix(audio)
        audio = zero_pad(audio, sr * 1) # we need the audio to be at least 1s

        feature_vector = preprocessor(audio, sr)

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

def save_confusion_matrix(m, labels, save_path):
    import plotly.figure_factory as ff

    x = labels
    y = labels

    # change each element of z to type string for annotations
    m_text = [[str(y) for y in x] for x in m]

    # set up figure 
    fig = ff.create_annotated_heatmap(m, x=x, y=y, annotation_text=m_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>')

    # # add custom xaxis title
    # fig.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=0.5,
    #                         y=-0.15,
    #                         showarrow=False,
    #                         text="Predicted value",
    #                         xref="paper",
    #                         yref="paper"))

    # # add custom yaxis title
    # fig.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=-0.35,
    #                         y=0.5,
    #                         showarrow=False,
    #                         text="Real value",
    #                         textangle=-90,
    #                         xref="paper",
    #                         yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.write_html(save_path)

def train(params):
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
    dataset = PhilharmoniaSet(path_to_dataset, classes, load_audio=False)
    train_loader, val_loader = train_test_split(
                                        dataset, 
                                        batch_size=1, 
                                        val_split=params['val_split'], 
                                        shuffle=True,
                                        random_seed=params['seed']
                                        )

    print(dataset.get_class_data())

    # --------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------

    # now, train our ised model
    train_features, train_labels, label_indices = get_features_and_labels(train_loader, 
                                    preprocessor=preprocessor,
                                    embedding_root=os.path.join('embeddings', '1s_chunks', params['preprocessor']))

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['fischer_reweighting']:
        print('doing fischer reweighting...')

        fischer_weights = get_fischer_weights(train_features, train_labels) 
        train_features = np.array([fischer_weights * f for f in train_features])
        dump(fischer_weights, f'{params["output_path"]}/{params["name"]}_fischer')

    # --------------------------------------------
    # CLASSIFIER SETUP AND PCA ON DATA
    # --------------------------------------------

    pca = PCA(n_components=params['pca_n_components'])
    X = pca.fit_transform(train_features)
    dump(pca, f'{params["output_path"]}/{params["name"]}_pca')

    classifier = load_classifier(params['classifier'])

    # fit our classifier
    classifier.fit(X, train_labels)
    dump(classifier, f'{params["output_path"]}/{params["name"]}')

    test_features, test_labels, label_indices = get_features_and_labels(val_loader,
                                    preprocessor=preprocessor,
                                    embedding_root=os.path.join('embeddings', '1s_chunks', params['preprocessor']))

    # --------------------------------------------
    # USE WEIGHTS??!??!!
    # --------------------------------------------
    if params['fischer_reweighting']:
        test_features = np.array([fischer_weights * f for f in test_features])

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
    save_confusion_matrix(m, label_indices, 
                    save_path=os.path.join(params["output_path"], 'confusion_matrix.html')
    )
    pd.DataFrame(m).to_csv(os.path.join(params["output_path"], 'confusion_matrix.csv'))
    pd.DataFrame(label_indices).to_csv(os.path.join(params["output_path"], 'label_indices.csv'))

    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(yt, yp)
    metrics['precision'] = sklearn.metrics.precision_score(yt, yp, average='macro')
    metrics['recall'] = sklearn.metrics.recall_score(yt, yp, average='macro')
    metrics['f1'] = sklearn.metrics.f1_score(yt, yp, average='macro')

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
                    out = train(params)
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