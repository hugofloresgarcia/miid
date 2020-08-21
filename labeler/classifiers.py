import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, 
                num_epochs, learning_rate,):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.SoftMax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

    def fit(self, X, y):

        