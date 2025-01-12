from os.path import exists, join
from typing import List

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from models.code2seq.path_context_dataset import PathContextDataset, PathContextSample
from models.code2seq.data_classes import PathContextBatch
from framework.utils.vocabulary import Vocabulary_c2s

class C2VPathContextDataModule:
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_c2s):
        self._config = config
        self._vocabulary = vocabulary

        self._dataset_dir = join(config.data_folder, config.name, config.dataset.name)
        self._train_data_file = join(self._dataset_dir, "train.c2v")
        self._val_data_file = join(self._dataset_dir, "val.c2v")
        self._test_data_file = join(self._dataset_dir, "test.c2v")

        # Check dataset existence
        if not exists(self._dataset_dir):
            raise ValueError(f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3 if not exists

    @staticmethod
    def collate_wrapper(batch: List[PathContextSample]) -> PathContextBatch:
        return PathContextBatch(batch)

    def _create_dataset(self, data_file: str, random_context: bool) -> Dataset:
        return PathContextDataset(data_file, self._config, self._vocabulary, random_context)

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self._train_data_file, self._config.hyper_parameters.random_context)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=self._config.hyper_parameters.shuffle_data,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self._val_data_file, False)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self._test_data_file, False)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True
        )

    def transfer_batch_to_device(self, batch: PathContextBatch, device: torch.device) -> PathContextBatch:
        batch.move_to_device(device)
        return batch


""" from os.path import exists, join
from typing import List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from models.code2seq.path_context_dataset import PathContextDataset, PathContextSample
from models.code2seq.data_classes import PathContextBatch
from utils.vocabulary import Vocabulary_c2s


class C2VPathContextDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_c2s):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary

        self._dataset_dir = join(config.data_folder, config.name,
                                 config.dataset.name)
        self._train_data_file = join(self._dataset_dir, "train.c2v")
        self._val_data_file = join(self._dataset_dir, "val.c2v")
        self._test_data_file = join(self._dataset_dir, "test.c2v")

    def prepare_data(self):
        if not exists(self._dataset_dir):
            raise ValueError(
                f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3 if not exists

    def setup(self, stage: Optional[str] = None):
        # TODO: collect or convert vocabulary if needed
        pass

    @staticmethod
    def collate_wrapper(batch: List[PathContextSample]) -> PathContextBatch:
        return PathContextBatch(batch)

    def _create_dataset(self, data_file: str, random_context: bool) -> Dataset:
        return PathContextDataset(data_file, self._config, self._vocabulary,
                                  random_context)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(
            self._train_data_file,
            self._config.hyper_parameters.random_context)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=self._config.hyper_parameters.shuffle_data,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._val_data_file, False)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._test_data_file, False)
        self.test_n_samples = dataset.get_n_samples()
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(self, batch: PathContextBatch,
                                 device: torch.device) -> PathContextBatch:
        batch.move_to_device(device)
        return batch
 """
