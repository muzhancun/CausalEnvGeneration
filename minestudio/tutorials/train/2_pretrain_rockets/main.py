'''
Date: 2024-11-24 08:23:02
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-26 06:29:11
FilePath: /MineStudio/minestudio/tutorials/train/2_pretrain_rockets/main.py
'''
import hydra
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.data import MineDataModule
from minestudio.train import MineLightning
from minestudio.models import RocketOnePolicy
from minestudio.train.callbacks import BehaviorCloneCallback
from minestudio.train.utils import convert_to_normal

logger = WandbLogger(project="minestudio")

@hydra.main(config_path='.', config_name='rocket_small')
def main(args):
    
    rocket_policy = RocketOnePolicy(
        backbone=args.model.backbone,
        hiddim=args.model.hiddim,
        num_heads=args.model.num_heads,
        num_layers=args.model.num_layers,
        timesteps=args.model.timesteps,
        mem_len=args.model.mem_len
    )
    
    mine_lightning = MineLightning(
        mine_policy=rocket_policy, 
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            BehaviorCloneCallback(weight=1.0),
        ]
    )
    
    mine_lightning.save_hyperparameters(convert_to_normal(args))

    mine_data = MineDataModule(
        data_params=dict(
            mode='raw',
            dataset_dirs=args.dataset_dirs,
            frame_width=224,
            frame_height=224,
            win_len=128,
            enable_segment=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_ratio=args.split_ratio, 
    )

    callbacks = []
    if args.save_freq > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="step-{step:06d}",
            every_n_train_steps=args.save_freq,
            save_top_k=-1,
        )
        callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger, 
        devices=args.devices, 
        precision=16, 
        strategy='ddp_find_unused_parameters_true', 
        use_distributed_sampler=False, 
    )
    trainer.fit(mine_lightning, datamodule=mine_data)

if __name__ == '__main__':
    main()