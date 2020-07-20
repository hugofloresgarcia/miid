import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import os
import numpy as np


def train_test_split(dataset, batch_size=1,
                     val_split=0.2, shuffle=True,
                     random_seed=42):
    """
    i stole this from
    https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    """
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=valid_sampler)

    return train_loader, val_loader


def debatch(data):
    """
     convert batch size 1 to None
    """
    data['audio'] = data['audio'].squeeze(dim=0)
    # convert str tuples to str
    data['sr'] = int(data['sr'])
    data['instrument'] = data['instrument'][0]
    data['pitch'] = data['pitch'][0]
    return data


class PhilharmoniaSet(Dataset):
    def __init__(self, path_to_csv, classes: tuple = None):
        super().__init__()

        assert os.path.exists(path_to_csv), f"couldn't find {path_to_csv}"
        # generate a list of dicts from our dataframe
        self.metadata = pd.read_csv(path_to_csv).to_dict('records')

        # remove all the classes not specified, unless it was left as None
        if classes:
            self.metadata = [e for e in self.metadata if e['instrument'] in classes]

        # class names (so many)
        self.classes = list(set([e['instrument'] for e in self.metadata]))

    def get_class_data(self):
        """
        return a tuple with unique class names and their number of items
        """
        classes = []
        for c in self.classes:
            subset = [e for e in self.metadata if e['instrument'] == c]
            info = (c, len(subset))
            classes.append(info)

        return tuple(classes)

    def retrieve_entry(self, entry):
        path_to_audio = entry['path_to_audio']

        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"
        # import our audio using torchaudio
        audio, sr = torchaudio.load(path_to_audio)

        instrument = entry['instrument']
        pitch = entry['pitch']

        data = {
            'audio': audio,
            'sr': sr,
            'instrument': instrument,
            'pitch': pitch
        }
        return data

    def __getitem__(self, index):
        def retrieve(index):
            return self.retrieve_entry(self.metadata[index])

        if isinstance(index, int):
            return retrieve(index)
        elif isinstance(index, slice):
            result = []
            start, stop, step = index.indices(len(self))
            for idx in range(start, stop, step):
                result.append(retrieve(idx))
            return result
        else:
            raise TypeError("index is neither an int or a slice")

    def get_example(self, class_name):
        subset = [e for e in self.metadata if e['instrument'] == class_name]
        # get a random index
        idx = torch.randint(0, high=len(subset), size=(1,)).item()

        entry = subset[idx]
        return self.retrieve_entry(entry)

    def __len__(self):
        return len(self.metadata)

    def get_onehot(self, label):
        return np.array([1 if label == l else 0 for l in self.classes])

