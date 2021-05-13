# -*- coding: utf-8 -*-
import argparse
from typing import Tuple, List, Dict, Union, Any

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from collab.model import EmbeddingNet


class LitNeuralCollaborativeFiltering(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(LitNeuralCollaborativeFiltering, self).__init__()
        self.save_hyperparameters()
        self.model = EmbeddingNet(
            n_users=self.hparams.n_users,
            n_items=self.hparams.n_items,
            n_factors=self.hparams.n_factors,
            embedding_dropout=self.hparams.embedding_dropout,
            hidden=self.hparams.hidden,
            dropouts=self.hparams.dropouts,
        )
        self.loss = MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.metrics = {"mae": self.mae, "mse": self.mse, "rmse": self._rmse}

    def _rmse(self, preds, targets):
        return torch.sqrt(self.mse(preds, targets))

    def forward(self, x):
        return self.model(x[:, 0], x[:, 1], self.hparams.minmax)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        log_dict = {
            "val/loss": loss,
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(y_hat, y)
            if metric_name == "rmse":
                log_dict["hp_metric"] = metric_value
            log_dict[f"val/{metric_name}"] = metric_value

        self.log_dict(log_dict, on_step=False, on_epoch=True)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, Any]]]]:
        optimizer = Adam(self.model.parameters(), 1e-1, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.hparams.max_epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(
            parents=[parent], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--max_lr",
            default=6e-4,
            type=float,
        )
        parser.add_argument(
            "--n_factors",
            default=50,
            type=int,
        )
        parser.add_argument(
            "--wd",
            default=1e-5,
            type=float,
        )
        parser.add_argument(
            "--momentum",
            default=0.9,
            type=float,
        )
        parser.add_argument(
            "--embedding_dropout",
            default=0.02,
            type=float,
        )
        parser.add_argument(
            "--hidden",
            default=[500, 500, 500],
            type=List[int],
        )
        parser.add_argument(
            "--dropouts",
            default=[0.2, 0.2, 0.2],
            type=List[float],
        )
        return parser
