# -*- coding: utf-8 -*-
import argparse
import logging
import warnings
from pprint import pprint
from typing import Union

import numpy as np
import pytorch_lightning as pl
from collab.ncf import LitNeuralCollaborativeFiltering
from collab.datamodule import CollabDataModule
from collab.utils import prepare_training

np.set_printoptions(precision=3)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


def parse_args(arguments: argparse.Namespace = None) -> argparse.Namespace:
    """
    Parses the program arguments.

    :param arguments: The argparse Namespace. Optional.
    :return: The Namespace with parsed parameters.
    """

    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CollabDataModule.add_data_specific_args(parser)
    parser = LitNeuralCollaborativeFiltering.add_model_specific_args(parser)

    # training config args
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--val_check_interval", type=Union[int, float], default=1.0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr_find_only", type=bool, default=False)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--print_config", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_model_path", type=str, default="./model_weights")
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--checkpoint_monitor_metric", type=str, default="hp_metric")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=5)
    parser.add_argument("--flush_logs_every_n_steps", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)

    # args for training from pre-trained model
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="A path to pre-trained model checkpoint. Required for fine tuning.",
    )
    parsed_arguments = parser.parse_args()
    parsed_arguments.experiment_name = arguments.experiment_name

    return parsed_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve", add_help=False)
    parser.add_argument("--experiment_name", type=str, default="gen-pre-training")

    arguments = parser.parse_args()
    arguments = parse_args(arguments)

    if arguments.print_config:
        print("Running with following configuration:")  # noqa T001
        pprint(vars(arguments))  # noqa T003

    pl.seed_everything(seed=arguments.seed)

    net, dm, trainer = prepare_training(arguments)

    if arguments.lr_find_only:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model=net, datamodule=dm, max_lr=1e-1)

        # Plot lr find results
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logging.info(f"LR Finder suggestion: {new_lr}")
    else:
        trainer.fit(model=net, datamodule=dm)
        if ~arguments.fast_dev_run:
            trainer.test()
