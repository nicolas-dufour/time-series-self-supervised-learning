from sklearn.manifold import TSNE
import pytorch_lightning as pl
from loss import TripletLossLogi, TripletLossXent
from model import CausalCNNEncoder
import plotly.express as px
import torch
import numpy as np
import pandas as pd


class TimeSeriesEmbedder(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        lr,
        weight_decay,
        betas,
        multivariate=False,
        loss="xent",
        temp=1,
    ):
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
        self.multivariate = multivariate
#         self.val_tsne_rep = pd.DataFrame(columns=["labels", "x", "y", "step"])
        if loss == "xent":
            self.criterium = TripletLossXent(temp=temp)
        elif loss == "logi":
            self.criterium = TripletLossLogi(temp=temp)
        else:
            raise "Loss not supported"

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        ref_series, pos_series = batch
        if self.multivariate:
            ref_emb = self(ref_series)
            pos_emb = self(pos_series)
        else:
            ref_emb = self(ref_series[:, None, :])
            pos_emb = self(pos_series[:, None, :])
        loss = self.criterium(ref_emb, pos_emb)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     labels, series = batch
    #     if self.multivariate:
    #         emb = self(series)
    #     else:
    #         emb = self(series[:, None, :])
    #     return {"labels": labels, "emb": emb.cpu()}

    # def validation_epoch_end(self, val_step_outputs):
    #     labels = val_step_outputs[0]["labels"].cpu().numpy()
    #     embs = val_step_outputs[0]["emb"].cpu().numpy()

    #     for i in range(1, len(val_step_outputs)):
    #         labels = np.concatenate((labels, val_step_outputs[i]["labels"].cpu().numpy()), axis=0)
    #         embs = np.concatenate((embs, val_step_outputs[i]["emb"].cpu().numpy()), axis=0)

    #     tsne = TSNE(n_components=2, random_state=21)
    #     projected_emb = pd.DataFrame(
    #         np.concatenate((labels[:, None], tsne.fit_transform(embs)), axis=1),
    #         columns=["labels", "x", "y"],
    #     )
    #     projected_emb["labels"] = projected_emb["labels"].astype("int").astype("str")
    #     projected_emb["step"] = self.global_step
    #     self.val_tsne_rep = pd.concat([self.val_tsne_rep, projected_emb])

    #     if self.global_step == self.trainer.max_steps:
    #         self.log(
    #             "TSNE_train_proj",
    #             px.scatter(
    #                 projected_emb, x="x", y="y", color="labels", animation_frame="step"
    #             ),
    #         )
    #         print("Ended")

    def test_step(self, batch, batch_idx):
        labels, series = batch
        if self.multivariate:
            emb = self(series)
        else:
            emb = self(series[:, None, :])
        return {"labels": labels, "emb": emb.cpu()}

    def test_epoch_end(self, val_step_outputs):
        labels = val_step_outputs[0]["labels"].cpu().numpy()
        embs = val_step_outputs[0]["emb"].cpu().numpy()

        for i in range(1, len(val_step_outputs)):
            labels = np.concatenate(
                (labels, val_step_outputs[i]["labels"].cpu().numpy()), axis=0
            )
            embs = np.concatenate(
                (embs, val_step_outputs[i]["emb"].cpu().numpy()), axis=0
            )

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
