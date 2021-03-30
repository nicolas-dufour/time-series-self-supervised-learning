from sklearn.manifold import TSNE
from sklearn.model_selection import (GridSearchCV,train_test_split, cross_val_score)
from sklearn.svm import SVC
import pytorch_lightning as pl
from loss import TripletLoss
from model import CausalCNNEncoder
from datamodule import UnivariateTestDataset
import plotly.express as px
import torch
import numpy as np
import pandas as pd


class TimeSeriesEmbedder(pl.LightningModule):
    def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, lr, weight_decay, betas, train_path, test_path):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = CausalCNNEncoder(
            in_channels=in_channels,
            channels=channels,
            depth=depth,
            reduced_size=reduced_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.criterium = TripletLoss()

        self.val_tsne_rep = pd.DataFrame(columns=["labels", "x", "y", "step"])

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        ref_series, pos_series = batch
        ref_emb = self(ref_series[:, None, :])
        pos_emb = self(pos_series[:, None, :])
        loss = self.criterium(ref_emb, pos_emb)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels, series = batch
        emb = self(series[:, None, :])
        return {"labels": labels, "emb": emb.cpu()}

    def validation_epoch_end(self, val_step_outputs):
        labels = val_step_outputs[0]["labels"]
        embs = val_step_outputs[0]["emb"]

        for i in range(1, len(val_step_outputs)):
            labels = np.concatenate((labels, val_step_outputs[i]["labels"]), axis=0)
            embs = np.concatenate((embs, val_step_outputs[i]["emb"]), axis=0)

        tsne = TSNE(n_components=2, random_state=21)
        projected_emb = pd.DataFrame(
            np.concatenate((labels[:, None], tsne.fit_transform(embs)), axis=1),
            columns=["labels", "x", "y"],
        )
        projected_emb["labels"] = projected_emb["labels"].astype("int").astype("str")
        projected_emb["step"] = self.global_step
        self.val_tsne_rep = pd.concat([self.val_tsne_rep, projected_emb])

        if self.global_step == self.trainer.max_steps:
            self.log(
                "TSNE_train_proj",
                px.scatter(
                    projected_emb, x="x", y="y", color="labels", animation_frame="step"
                ),
            )
            print("Ended")

    def test_step(self, batch, batch_idx):
        labels, series = batch
        emb = self(series[:, None, :])
        return {"labels": labels, "emb": emb.cpu()}

    def test_epoch_end(self, val_step_outputs):
        labels = val_step_outputs[0]["labels"]
        embs = val_step_outputs[0]["emb"]

        for i in range(1, len(val_step_outputs)):
            labels = np.concatenate((labels, val_step_outputs[i]["labels"]), axis=0)
            embs = np.concatenate((embs, val_step_outputs[i]["emb"]), axis=0)

        tsne = TSNE(n_components=2, random_state=21)
        projected_emb = pd.DataFrame(
            np.concatenate((labels[:, None], tsne.fit_transform(embs)), axis=1),
            columns=["labels", "x", "y"],
        )
        projected_emb["labels"] = projected_emb["labels"].astype("int").astype("str")
        self.log(
            "TSNE_test_proj", px.scatter(projected_emb, x="x", y="y", color="labels")
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )
        return optimizer
    
    def compute_scores(self):
        '''
            Compute SVM accuracy score on the train then test set. Is sufficient train data, SVM C hyperparameters is found using grid search.  
        '''
        
        train_set = UnivariateTestDataset(self.train_path, fill_na=True)
        train_emb = self(torch.Tensor(train_set.time_series[:,None,:])).detach().numpy()
        train_labels = train_set.labels
        
        nb_classes = len(np.unique(train_labels))
        train_size = train_emb.shape[0]

        classifier = SVC(C=np.inf, gamma='scale')
        if train_size // nb_classes < 5 or train_size < 50:
            classifier = classifier.fit(train_emb, train_labels)
        else:
            grid_search = GridSearchCV(classifier, {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,np.inf],
                                                                            'kernel': ['linear'],'degree': [3],'gamma': ['scale'],'coef0': [0],'shrinking': [True],'probability': [False],'tol': [0.001],
                                                                             'cache_size': [200], 'class_weight': [None],'verbose': [False],'max_iter': [10000000],
                                                                            'decision_function_shape': ['ovr'],'random_state': [None]}, cv=5, n_jobs=5)
            if train_size <= 10000:
                grid_search.fit(train_emb, train_labels)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = train_test_split(
                    train_emb, train_labels,
                    train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            classifier = grid_search.best_estimator_

        # Cross validation score
        self.train_score = np.mean(cross_val_score(classifier, train_emb, y=train_labels, cv=5, n_jobs=5))
        
        
        # Predict class for the test set
        test_set = UnivariateTestDataset(self.test_path, fill_na=True)
        test_emb = self(torch.Tensor(test_set.time_series[:,None,:])).detach().numpy()
        test_labels = test_set.labels
        
        self.test_score = np.mean(cross_val_score(classifier, test_emb, y=test_labels, cv=5, n_jobs=5))

        
        return self.train_score, self.test_score



