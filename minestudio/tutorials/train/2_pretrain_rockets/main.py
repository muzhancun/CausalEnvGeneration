'''
Date: 2024-11-24 08:23:02
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:25:02
FilePath: /MineStudio/minestudio/tutorials/train/rocket_one.py
'''
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.data import MineDataModule
from minestudio.train import MineLightning
from minestudio.models import load_openai_policy
from minestudio.train.callbacks import BehaviorCloneCallback

# policy = 

mine_lightning = MineLightning(
    mine_policy=load_openai_policy(
        model_path='/nfs-shared/jarvisbase/pretrained/foundation-model-1x.model',
        weights_path='/nfs-shared/jarvisbase/pretrained/foundation-model-1x.weights',
    ),
    log_freq=20,
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    callbacks=[
        BehaviorCloneCallback(weight=1.0),
    ]
)

mine_data = MineDataModule(
    data_params=dict(
        mode='raw',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_6xx',
            '/nfs-shared-2/data/contractors/dataset_7xx',
            '/nfs-shared-2/data/contractors/dataset_8xx',
            '/nfs-shared-2/data/contractors/dataset_9xx',
            '/nfs-shared-2/data/contractors/dataset_10xx',
        ],
        frame_width=128,
        frame_height=128,
        win_len=128,
    ),
    batch_size=8,
    num_workers=8,
    prefetch_factor=4,
    split_ratio=0.9, 
)

wandb_logger = None
wandb_logger = WandbLogger(project="minestudio", id="new_tune_vpt_fd_1x_short_mem")
trainer = L.Trainer(logger=wandb_logger, devices=8, precision=16, strategy='ddp_find_unused_parameters_true', use_distributed_sampler=False)
trainer.fit(mine_lightning, datamodule=mine_data)