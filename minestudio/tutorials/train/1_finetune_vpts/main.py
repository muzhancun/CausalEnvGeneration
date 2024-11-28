'''
Date: 2024-11-12 14:00:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 16:12:24
FilePath: /MineStudio/minestudio/tutorials/train/1_finetune_vpts/main.py
'''
import hydra
import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from typing import Dict, Any, Tuple

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from minestudio.data import MineDataModule
from minestudio.train import MineLightning
from minestudio.models import load_openai_policy
from minestudio.train.utils import convert_to_normal
from minestudio.train.mine_callbacks import BehaviorCloneCallback
from minestudio.train.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback

logger = WandbLogger(project="minestudio")

@hydra.main(config_path='.', config_name='vpt_config')
def main(args):
    
    mine_lightning = MineLightning(
        mine_policy=load_openai_policy(
            model_path=args.model_path,
            weights_path=args.weights_path,
        ),
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            BehaviorCloneCallback(weight=1.0),
        ], 
        hyperparameters=convert_to_normal(args),
    )

    mine_data = MineDataModule(
        data_params=dict(
            mode='raw',
            dataset_dirs=args.dataset_dirs,
            frame_width=128,
            frame_height=128,
            win_len=128,
        ),
        train_shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_ratio=args.split_ratio, 
    )

    L.Trainer(
        logger=logger, 
        devices=args.devices, 
        precision=16, 
        strategy='ddp_find_unused_parameters_true', 
        use_distributed_sampler=False, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            SpeedMonitorCallback(),
            SmartCheckpointCallback(
                dirpath='./weights', filename='weight-{epoch}-{step}', save_top_k=-1, 
                every_n_train_steps=args.save_freq, save_weights_only=True,
            ), 
            SmartCheckpointCallback(
                dirpath='./checkpoints', filename='ckpt-{epoch}-{step}', save_top_k=1, 
                every_n_train_steps=args.save_freq+1, save_weights_only=False,
            )
        ]
    ).fit(
        model=mine_lightning, 
        datamodule=mine_data
    )

if __name__ == '__main__':
    main()