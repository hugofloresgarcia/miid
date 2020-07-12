import numpy as np
import torch


def distance(a, b):
    """
    euclidean distance
    """
    return np.sqrt(torch.sum((a - b) ** 2))


def _arg_get_neighbors(num_neighbors: int, sample: torch.Tensor, examples: torch.Tensor):
    if len(examples) == 0:
        return None
    # compute distances
    distances = []
    for e in examples:
        distances.append(distance(e, sample))

    # get the nearest neighbor's index
    idxs = np.argsort(np.array(distances))

    return idxs[-1:(-1 - num_neighbors)]


def get_neighbors(num_neighbors: int, sample: torch.Tensor, examples: torch.Tensor):
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
        return self.labels[idx]
