'''
Date: 2024-11-12 14:00:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-12 13:22:12
FilePath: /MineStudio/minestudio/tutorials/train/finetune_vpt.py
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

mine_lightning = MineLightning(
    mine_policy=load_openai_policy(
        model_path='/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model',
        weights_path='/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights',
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
            '/nfs-shared-2/data/contractors/dataset_10xx',
        ],
        frame_width=128,
        frame_height=128,
        win_len=128,
    ),
    batch_size=4,
    num_workers=8,
    prefetch_factor=4,
)

wandb_logger = WandbLogger(project="minestudio", id="tune_vpt_fd_2x")
trainer = L.Trainer(logger=wandb_logger, max_epochs=1, devices=1, precision=16, strategy='ddp_find_unused_parameters_true', use_distributed_sampler=False)
trainer.fit(mine_lightning, datamodule=mine_data)