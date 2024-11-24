'''
Date: 2024-11-12 14:00:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:46:19
FilePath: /MineStudio/minestudio/tutorials/train/1_finetune_vpts/main.py
'''
import hydra
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

@hydra.main(config_path='.', config_name='vpt_2x')
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
        ]
    )

    mine_data = MineDataModule(
        data_params=dict(
            mode='raw',
            dataset_dirs=args.dataset_dirs,
            frame_width=128,
            frame_height=128,
            win_len=128,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_ratio=args.split_ratio, 
    )

    if args.logger == 'wandb':
        logger = WandbLogger(project="minestudio", id=args.project_id)
    else:
        logger = None
    trainer = L.Trainer(logger=logger, devices=args.devices, precision=16, strategy='ddp_find_unused_parameters_true', use_distributed_sampler=False)
    trainer.fit(mine_lightning, datamodule=mine_data)

if __name__ == '__main__':
    main()