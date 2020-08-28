import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import os
import sklearn

import labeler
from philharmonia_dataset import train_test_split

import embeddings.embedding_dataset as embedding_dataset

from test_tube import Experiment

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def collate_embeddings(batch):
    print(f'LENGTH OF BATCH: {len(batch)}')
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
        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 19)

        self.hparams = hparams
        self.splits = np.array([0.7, 0.15, 0.15])
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.dropout_p = hparams.dropout

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',    default=4,      type=int)
        parser.add_argument('--learning_rate',  default=1e-4,   type=float)
        parser.add_argument('--batch_size',     default=64,    type=int)
        parser.add_argument('--dropout',        default=0.5,    type=int)
        parser.add_argument('--mixup',          default=True,   type=str2bool)
        return parser

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = torch.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = torch.log_softmax(self.fc4(x), dim=1)
  
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def preprocess(self, X, y=None):
        # preprocessing steps

        # standardize
        if X.shape[0] > 2:
            X = (X - X.mean(dim=0)) / X.std(dim=0)
        
        return X

    def training_step(self, batch, batch_idx):
        X, y = batch['embedding'], batch['one_hot']

        if X.shape[0] < self.batch_size:
            print("SOMETHING FISHY: X")
            print(X)
            print("SOMETHING FISHY: y")
            print(y)

        
        mixup = hparams.mixup
        X = self.preprocess(X)

        # mixup would go here
        if mixup:
            mixed_x, y_a, y_b, lam = mixup_data(X, y, alpha=1.0)
            X = mixed_x
            y_a = torch.argmax(y_a, dim=1, keepdim=False)
            y_b = torch.argmax(y_b, dim=1, keepdim=False)

        # for nll loss
        y = torch.argmax(y, dim=1, keepdim=False)
        
        logits = self.forward(X) 

        if X.shape[0] < self.batch_size:
            print("SOMETHING FISHY: X")
            print(X)
            print("SOMETHING FISHY: y")
            print(y)
            print(f'SOMETHING FISHY: LOGITS {logits}')
        
        
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
        result.log('val_accuracy', accuracy)
        result.log('val_f1', f1)
        result.log('val_loss',  loss)

        result.ytrue = y
        result.yhat = yhat
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

        result.ytrue = y
        result.yhat = yhat
        return result

    def _log_epoch_metrics(self, outputs, prefix='val'):
        # calculate confusion matrices
        y, yhat = outputs.ytrue, outputs.yhat
        outputs.ytrue = torch.Tensor(0)
        outputs.yhat = torch.Tensor(0)

        conf_matrix = sklearn.metrics.confusion_matrix(yhat.detach().numpy(), y.detach().numpy(), labels=list(range(len(self.dataset.classes))))
        norm_conf_matrix = sklearn.metrics.confusion_matrix(yhat.detach().numpy(), y.detach().numpy(), labels=list(range(len(self.dataset.classes))), normalize='true')
        norm_conf_matrix = np.around(norm_conf_matrix, 3)

        # get plotly images as byte array
        conf_matrix = labeler.utils.save_confusion_matrix(conf_matrix, labels=self.dataset.classes, save_path='.numpy')
        norm_conf_matrix = labeler.utils.save_confusion_matrix(norm_conf_matrix, labels=self.dataset.classes, save_path='.numpy')

        # log images
        self.logger.experiment.add_image(f'{prefix}_conf_matrix', conf_matrix, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image(f'{prefix}_conf_matrix_normalized', norm_conf_matrix, self.current_epoch, dataformats='HWC')

        for metric in outputs.epoch_log_metrics:
            outputs[metric] = outputs[metric].mean()

        return outputs

    def validation_epoch_end(self, outputs):
        outputs = self._log_epoch_metrics(outputs, prefix='val')
        return outputs

    def test_epoch_end(self, outputs):
        outputs = self._log_epoch_metrics(outputs, prefix='test')
        return outputs

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
        #  only overriding this for now
        self.train_loader, self.val_loader = train_test_split(self.dataset, batch_size=self.batch_size, val_split=0.3, shuffle=True, random_seed=42, collate_fn=collate_embeddings)

    def train_dataloader(self):
        return self.train_loader #  only overriding this for now
        return DataLoader(self.train_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):  
        return self.val_loader # only overriding this for now
        return DataLoader(self.val_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, 
            collate_fn=collate_embeddings, shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # return [optimizer], [scheduler]
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
        checkpoint_callback=checkpoint_callback, 
        logger=logger,
        terminate_on_nan=True,
        # gradient_clip_val=0.5,
        # progress_bar_refresh_rate=0,
        # callbacks=callbacks
        # resume_from_checkpoint='./logs/two-hidden-mixup-lr-3/checkpoints/epoch=12-val_loss=0.28.ckpt'
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

    parser.add_argument('--num_epochs',     default=70,    type=int)
    parser.add_argument('--name',           default='exp', type=str)
    parser.add_argument('--random_seed',    default=42,   type=int)
    parser.add_argument('--verbose',        default=True,  type=str2bool)


    hparams = parser.parse_args()

    train(hparams)