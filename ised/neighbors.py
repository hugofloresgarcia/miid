import numpy as np
import torch
from statistics import mode


def distance(a, b):
    """
    euclidean distance
    """
    return np.sqrt(np.sum((a - b) ** 2))


def _arg_get_neighbors(num_neighbors: int, sample: np.ndarray, examples: np.ndarray):
    """
    get index for nearest neighbors
    """
    if len(examples) == 0:
        return None
    # compute distances
    distances = []
    for e in examples:
        distances.append(distance(e, sample))

    # get the nearest neighbor's index
    idxs = np.argsort(np.array(distances))

    return idxs[0:num_neighbors]


def get_neighbors(num_neighbors: int, sample: np.ndarray, examples: np.ndarray):
    """
    get nearest neighbors
    """
    idxs = _arg_get_neighbors(num_neighbors, sample, examples)
    # the nearest neighbor will be the last index
    return examples[idxs]


class KNN:
    def __init__(self, data, labels):
        """
        data: neighborhood to consider. must be a list of dicts,
        the dicts must have a 'label' and a 'features' key
        """
        self.data = data
        self.labels = labels

    def predict(self, x, num_neighbors):
        idx = _arg_get_neighbors(num_neighbors, x, self.data)
        # return the most popular neighbor
        return mode(self.labels[idx])
