'''
Date: 2024-11-12 14:00:50
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 14:21:59
FilePath: /MineStudio/scratch/caishaofei/workspace/MineStudio/minestudio/train/example.py
'''
import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from minestudio.data import MineDataModule
from minestudio.train import MineLightning
from minestudio.models import MinePolicy

class SimplePolicy(MinePolicy):

    def __init__(self, hiddim: int):
        super().__init__(hiddim=hiddim)
        self.net = nn.Sequential(
            nn.Linear(128*128*3, hiddim), 
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
        )

    def forward(self, input, state_in):
        x = rearrange(input['image'], 'b t h w c -> b t (h w c)')
        return self.net(x), state_in

    def initial_state(self, batch_size):
        return None

mine_lightning = MineLightning(
    mine_policy=SimplePolicy(hiddim=1024),
    log_freq=10,
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
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
    batch_size=2,
    num_workers=4,
    prefetch_factor=2
)

trainer = L.Trainer(max_epochs=1, devices=1, precision=16, strategy='ddp', use_distributed_sampler=False)

trainer.fit(mine_lightning, datamodule=mine_data)