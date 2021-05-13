# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging


class CollabDataset(Dataset):
    def __init__(self, X, y):
        super(CollabDataset, self).__init__()
        X, y = X.values, y.values
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CollabDataModule(LightningDataModule):
    def __init__(self, df_path, ptc_val=0.2, seed=42, batch_size=2048, num_workers=4):
        super(CollabDataModule, self).__init__()
        self.df_path = df_path
        self.ptc_val = ptc_val
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df = pd.read_csv(df_path)
        (
            (self.n_users, self.n_items),
            (X, y),
            (self.user_to_index, self.item_to_index),
        ) = self.prepare_data(self.df)

        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(
            X, y, test_size=ptc_val, random_state=seed
        )
        self.minmax = y.min().astype(float), y.max().astype(float)

        logging.info(f"Embeddings: {self.n_users} users, {self.n_items} items")
        logging.info(f"Dataset shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Min-Max: {self.minmax}")

    def prepare_data(self, df):
        unique_users = df["USER_ID"].unique()
        user_to_index = {old: new for new, old in enumerate(unique_users)}
        new_users = df["USER_ID"].map(user_to_index)

        unique_items = df["ITEM_ID"].unique()
        item_to_index = {old: new for new, old in enumerate(unique_items)}
        new_items = df["ITEM_ID"].map(item_to_index)

        n_users = unique_users.shape[0]
        n_items = unique_items.shape[0]

        X = pd.DataFrame({"USER_ID": new_users, "ITEM_ID": new_items})
        y = df["EVENT_VALUE"].astype(np.float32)
        return (n_users, n_items), (X, y), (user_to_index, item_to_index)

    def train_dataloader(self):
        train_split = CollabDataset(self.X_train, self.y_train)
        return DataLoader(
            train_split,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_split = CollabDataset(self.X_test, self.y_test)
        return DataLoader(
            val_split,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def inference_dataloader(self, item_ids, user_id=None):
        pass

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds datamodule specific arguments.

        :param parent_parser: The parent parser.
        :returns: The parser.
        """
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--df_path",
            type=str,
            default="./datasets/ratings.csv",
        )
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--ptc_val", type=float, default=0.2)
        return parser


if __name__ == "__main__":
    dm = CollabDataModule("../datasets/ratings.csv")
    dl = dm.val_dataloader()
    for idx, batch in enumerate(dl):
        u, i, r = batch
        print(idx, u.shape, i.shape, r.shape)  # noqa T001
        break
