import matplotlib.pyplot as plt
import numpy as np
from . import utils
plt.ioff()


def get_figure(n_rows, n_cols, title=None):
    """
    a wrapper around plt.figure
    to minimize imports i guess?
    """
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)

    axes = fig.subplots(n_rows, n_cols)
    return fig, axes

def plot_pca(ax, classes: dict, title=None):
    """
    pca scatterplot of labeled classes
    """


    if title is not None:
        ax.set_title(title)

    for label, data in classes.items():
        ax.scatter(data[:, 0], data[:, 1], label=label)

    ax.legend()

    return ax

def plot_features(ax, feature_vector: np.array, title='mfcc features'):
    fv = utils.assert_numpy(feature_vector)

    plt.sca(ax)
    plt.title(title)
    plt.imshow(fv.reshape(-1, 4))

    plt.xticks([0, 1, 2, 3],
               ['mean', 'var', 'delta_mean', 'delta_var'],
               rotation=90)
    plt.colorbar()


def plot_audio(ax, audio: np.array, sr=None):
    audio = utils.assert_numpy(audio)

    a = audio.reshape((-1,))
    t = np.arange(len(a))

    xlabel = ''
    if sr:
        pass
    else:
        xlabel = 'samples'


    ax.plot(t, a)
    ax.set_xlabel(xlabel)