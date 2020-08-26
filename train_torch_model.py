import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import os

import embeddings.embedding_dataset as embedding_dataset

def collate_embeddings(batch):
    # TODO: get rid of this spaghetti doodoo
    embeddings = np.concatenate([b['embedding'] for b in batch], axis=0)
    embeddings = torch.Tensor(embeddings)
    
    labels = []
    for embedding, label in zip([b['embedding'] for b in batch], [b['one_hot'] for b in batch]):
        sub_batch = embedding.shape[0]
        labels.append(torch.cat([torch.Tensor([label]) for _ in range(sub_batch)], dim=0))
    
    data = dict(
        embedding=embeddings, 
        one_hot=torch.cat(labels, dim=0)
    )
    return data

def mixup_data(x, y, alpha=1.0):
    """ return mixed inputs, targets, and lambda"""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class NN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.fc1 = nn.Linear(6144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 19)

        self.hparams = hparams
        self.splits = np.array([0.6, 0.2, 0.2])
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.dropout_p = hparams.dropout

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',    default=4, type=int)
        parser.add_argument('--learning_rate',  default=1e-3)
        parser.add_argument('--batch_size',     default=128)
        parser.add_argument('--dropout',        default=0.1)
        return parser

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def preprocess(self, X, y=None):
        # preprocessing steps

        # standardize
        X = (X - X.mean(dim=0)) / X.std(dim=0)
        
        return X

    def training_step(self, batch, batch_idx):
        X, y = batch['embedding'], batch['one_hot']
        
        mixup = True
        # mixup would go here
        if mixup:
            mixed_x, y_a, y_b, lam = mixup_data(X, y, alpha=1.0)
            X = mixed_x

        # for nll loss
        y = torch.argmax(y, dim=1, keepdim=False)
        y_a = torch.argmax(y_a, dim=1, keepdim=False)
        y_b = torch.argmax(y_b, dim=1, keepdim=False)

        X = self.preprocess(X)

        logits = self.forward(X) 
        loss = self.cross_entropy_loss(logits, y) if not mixup else mixup_criterion(logits, y_a, y_b, lam, criterion=self.cross_entropy_loss)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss,  on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        X, y = batch['embedding'], batch['one_hot']
        y = torch.argmax(y, dim=1, keepdim=False) # for nll loss

        X = self.preprocess(X)
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        yhat = torch.argmax(logits, dim=1, keepdim=False)
        accuracy = pl.metrics.functional.accuracy(yhat, y)
        f1 = pl.metrics.functional.f1_score(yhat, y)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('val_accuracy', accuracy, on_epoch=True, on_step=False, )
        result.log('val_f1', f1, on_epoch=True, on_step=False, )
        result.log('val_loss',  loss)
        return result
    
    def test_step(self, batch, batch_idx):
        X, y = batch['embedding'], batch['one_hot']
        y = torch.argmax(y, dim=1, keepdim=False)

        X = self.preprocess(X)
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        yhat = torch.argmax(logits, dim=1, keepdim=False)
        accuracy = pl.metrics.functional.accuracy(yhat, y)
        f1 = pl.metrics.functional.f1_score(yhat, y)

        result = pl.EvalResult()
        result.log('test_accuracy', accuracy, on_epoch=True, on_step=False)
        result.log('test_f1', f1, on_epoch=True, on_step=False, )
        result.log('test_loss', loss, on_step=True, on_epoch=False)
        return result

    def prepare_data(self):
        self.dataset = embedding_dataset.PhilharmoniaEmbeddings(
            dataset_path=os.path.join(os.getcwd(), 'embeddings',
                     'silence_split', 'openl3-mel256-6144-music'), 
            classes='no_percussion')

        lengths = (len(self.dataset)*self.splits).astype(int)
        # oof. hackiest thing ever.
        while sum(lengths) < len(self.dataset):
            lengths[-1] +=1 

        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
                        dataset=self.dataset, 
                        lengths=lengths)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):  
        return DataLoader(self.val_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def train(hparams):

    pl.seed_everything(hparams.random_seed)

    model = NN(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'logs', hparams.name, 'checkpoints', '{epoch}-{val_loss:.2f}'), 
        monitor='val_loss', 
        verbose=hparams.verbose, 
        save_top_k=1, 
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir='./logs', 
        name=hparams.name, 
        version=0
    )

    trainer = pl.Trainer.from_argparse_args(
        args=hparams,
        default_root_dir=os.path.join(os.getcwd(), 'logs'),
        early_stop_callback=True, 
        checkpoint_callback=checkpoint_callback, 
        logger=logger,
    )

    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # ------------------
    # | HYPERPARAMETERS |
    # ------------------

    parser = NN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--num_epochs',     default=70)
    parser.add_argument('--name',           default='exp', type=str)
    parser.add_argument('--random_seed',    default=420)
    parser.add_argument('--verbose',        default=True, type=bool)

    hparams = parser.parse_args()

    train(hparams)