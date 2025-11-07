# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Callable, Optional

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

from tsfm.common import hydra_util  # noqa: hydra resolvers
from tsfm.data.loader import DataLoader



class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
        data_builder=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.data_builder = data_builder

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    @staticmethod
    def get_torch_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
        )
    
    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_torch_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )


@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    trainer: L.Trainer = instantiate(cfg.trainer)
    
    # Instantiate the data builder and create dataset
    data_builder = instantiate(cfg.data)
    train_dataset: Dataset = data_builder.load_dataset(model.train_transform_map)
    
    val_dataset: Optional[Dataset | list[Dataset]] = (
        tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )
        if "val_data" in cfg
        else None
    )
    L.seed_everything(cfg.seed, workers=True)
    print("train_dataset:", train_dataset)
    print("val_dataset:", val_dataset)
    print("train_dataset size:", len(train_dataset))
    # print("val_dataset size:", len(val_dataset[0]), len(val_dataset[1]), len(val_dataset[2]))
    print("val_dataset size:", len(val_dataset[0]))
    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset, data_builder),  # Pass data_builder
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    main()
