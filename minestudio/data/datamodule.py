'''
Date: 2024-11-10 12:31:33
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 06:41:44
FilePath: /MineStudio/minestudio/data/datamodule.py
'''

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from concurrent.futures import ThreadPoolExecutor

from rich import print
from rich.console import Console
from collections import OrderedDict

from typing import Dict, List, Union, Sequence, Mapping, Any, Optional, Literal
import lightning.pytorch as pl

from minestudio.data.minecraft.dataset import MinecraftDataset
from minestudio.data.minecraft.utils import MineDistributedBatchSampler, batchify

class MineDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_params: Dict, 
        batch_size: int = 1,
        num_workers: int = 0,
        train_shuffle: bool = False,
        prefetch_factor: Optional[int] = None,
        **kwargs, 
    ):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.prefetch_factor = prefetch_factor
        self.kwargs = kwargs
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MinecraftDataset(split='train', **self.data_params, **self.kwargs)
        self.val_dataset   = MinecraftDataset(split='val', **self.data_params, **self.kwargs)

    def train_dataloader(self):
        if self.data_params['mode'] == 'raw':
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.train_dataset, 
                batch_size=self.batch_size, 
            )
            train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_sampler=batch_sampler,
                num_workers=self.num_workers, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=self.train_shuffle, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        return train_loader

    def val_dataloader(self):
        if self.data_params['mode'] == 'raw':
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.val_dataset, 
                batch_size=self.batch_size, 
            )
            val_loader = DataLoader(
                dataset=self.val_dataset, 
                batch_sampler=batch_sampler,
                num_workers=self.num_workers, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            val_loader = DataLoader(
                dataset=self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        return val_loader

if __name__ == '__main__':
    import lightning as L
    from tqdm import tqdm
    fabric = L.Fabric(accelerator="cuda", devices=8, strategy="ddp")
    fabric.launch()
    data_module = MineDataModule(
        data_params=dict(
            mode='raw',
            dataset_dirs=[
                '/nfs-shared-2/data/contractors/dataset_10xx',
            ],
            enable_contractor_info=False,
            enable_segment=True,
            frame_width=224,
            frame_height=224,
            win_len=128,
            skip_frame=1,
            split_ratio=0.2,
        ),
        batch_size=8,
        num_workers=8,
        train_shuffle=True,
        prefetch_factor=4,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    rank = fabric.local_rank
    for idx, batch in enumerate(tqdm(train_loader, disable=True)):
        print(
            f"{rank = } \t" + "\t".join(
                [f"{a[-20:]} {b}" for a, b in zip(batch['episode'], batch['progress'])]
            )
        )