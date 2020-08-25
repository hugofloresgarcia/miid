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

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class NN(pl.LightningModule):

    def __init__(self, 
                num_epochs, 
                learning_rate=1e-3,
                random_seed=42,
                train_val_test_split=np.array([0.6, 0.2, 0.2]),
                batch_size=128, 
                dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(6144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 19)

        self.seed = random_seed
        self.num_epochs = num_epochs
        self.splits = train_val_test_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_p = dropout

        pl.seed_everything(random_seed)

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
        # for nll loss
        y = torch.argmax(y, dim=1, keepdim=False)

        X = self.preprocess(X)

        # mixup would go here

        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result


    def validation_step(self, batch, batch_idx):
        X, y = batch['embedding'], batch['one_hot']
        y = torch.argmax(y, dim=1, keepdim=False) # for nll loss

        X = self.preprocess(X)
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        yhat = torch.argmax(logits, dim=1, keepdim=False)
        accuracy = pl.metrics.functional.accuracy(yhat, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_accuracy', accuracy, on_epoch=True, on_step=False, prog_bar=True)
        result.log('val_loss', loss, prog_bar=True)
        return result

    def prepare_data(self):
        self.dataset = embedding_dataset.PhilharmoniaEmbeddings(
            dataset_path=os.path.join(os.getcwd(), 'embeddings',
                     'silence_split', 'openl3-mel256-6144-music'), 
            classes='no_percussion')

        print(len(self.dataset))
        lengths = (len(self.dataset)*self.splits).astype(int)
        # oof. hackiest thing ever.
        while sum(lengths) < len(self.dataset):
            lengths[-1] +=1 

        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
                        dataset=self.dataset, 
                        lengths=lengths)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=collate_embeddings)

    def val_dataloader(self):  
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=collate_embeddings)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=collate_embeddings)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ------------------
# | HYPERPARAMETERS |
# ------------------

# MODEL ARGS
num_epochs = 500
learning_rate = 1e-3
random_seed = 42
train_split = 0.6
val_split = 0.2
test_split = 0.2
batch_size = 128

def train(
            # MODEL ARGS
            num_epochs=num_epochs, 
            learning_rate=learning_rate,
            random_seed=random_seed,
            train_val_test_split=np.array([0.6, 0.2, 0.2]),
            batch_size=batch_size, 
 
            verbose=True):

    model = NN(
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        random_seed=random_seed,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta='1e-5', 
        patience=5, 
        verbose=verbose,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints'), 
        monitor='val_loss', 
        verbose=verbose
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'logs'),
        early_stop_callback=early_stopping, 
        checkpoint_callback=checkpoint_callback, 
        fast_dev_run=False)

    # if os.path.exists(checkpoint_dir):
    #     print(f'loading checkpoint: {checkpoint_dir}')
    #     trainer =  pl.Trainer(resume_from_checkpoint=checkpoint_dir)

    trainer.fit(model)

if __name__ == "__main__":
    train(
        # MODEL ARGS
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        random_seed=random_seed,
        train_val_test_split=np.array([train_split, val_split, test_split]),
        batch_size=batch_size, 

        verbose=True
    )