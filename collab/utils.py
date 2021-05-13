# -*- coding: utf-8 -*-
import argparse
from typing import Tuple

import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from collab.ncf import LitNeuralCollaborativeFiltering
from collab.datamodule import CollabDataModule


def prepare_pl_module(args: argparse.Namespace) -> pl.LightningModule:
    """
    Prepares the Lightning Module.

    :param args: The arguments.
    :return: The Lightning Module.
    """

    net = LitNeuralCollaborativeFiltering(**vars(args))

    return net


def prepare_pl_trainer(args: argparse.Namespace) -> pl.Trainer:
    """
    Prepares the Pytorch Lightning Trainer.

    :param args: The arguments.
    :return: The Pytorch Lightning Trainer.
    """
    experiment_name = "ncf"
    tb_logger = pl_loggers.TensorBoardLogger(
        args.log_dir,
        name=experiment_name,
        default_hp_metric=False,
    )
    monitor_metric = args.checkpoint_monitor_metric
    mode = "min"
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stopping_patience,
        verbose=False,
        mode=mode,
    )
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        verbose=False,
        mode=mode,
        dirpath=args.save_model_path,
        filename=f"{experiment_name}-{{epoch:02d}}-{{step:05d}}-{{{monitor_metric}:.5f}}",
        save_top_k=args.save_top_k,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, early_stop_callback]
    checkpoint_callback = model_checkpoint
    pl_trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback,
    )
    return pl_trainer


def prepare_pl_datamodule(args: argparse.Namespace) -> pl.LightningDataModule:
    """
    Prepares the Tabular Lightning Data Module.

    :param args: The arguments.
    :return: The Tabular Lightning Data Module.
    """
    data_module = CollabDataModule(
        df_path=args.df_path,
        ptc_val=args.ptc_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return data_module


def prepare_training(
    args: argparse.Namespace,
) -> Tuple[pl.LightningModule, pl.LightningDataModule, pl.Trainer]:
    """
    Prepares everything for training. `DataModule` is prepared by setting up the train/val/test sets for specified fold.
    Creates new `PreTrainingESRGANModule` Lightning Module together with `pl.Trainer`.

    :param args: The arguments.
    :returns: A tuple with model and the trainer.
    """

    pl.seed_everything(args.seed)
    data_module = prepare_pl_datamodule(args)

    n_users = data_module.n_users
    n_items = data_module.n_items
    args.minmax = data_module.minmax

    args.n_users = n_users
    args.n_items = n_items

    lightning_module = prepare_pl_module(args)
    pl_trainer = prepare_pl_trainer(args)

    return lightning_module, data_module, pl_trainer
