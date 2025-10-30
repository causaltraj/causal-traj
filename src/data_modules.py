import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        train_bsz: int = 32,
        val_bsz: int = 32,
        train_size: int = None,
        val_size: int = None,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.train_bsz = train_bsz
        self.val_bsz = val_bsz
        self.train_size = train_size
        self.val_size = val_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data = np.load(self.data_dir / "train.npy", allow_pickle=True)
        test_data = np.load(self.data_dir / "test.npy", allow_pickle=True)
        if self.train_size is not None:
            train_data = train_data[: self.train_size]
        if self.val_size is not None:
            test_data = test_data[: self.val_size]
        self.dataset = {
            "train": train_data,
            "test": test_data,
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True, bsz=self.train_bsz)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False, bsz=self.val_bsz)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False, bsz=self.val_bsz)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, bsz: int = None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=bsz,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
