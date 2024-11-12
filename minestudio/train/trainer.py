'''
Date: 2024-11-10 13:44:13
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 17:21:53
FilePath: /MineStudio/minestudio/train/trainer.py
'''
import torch
import torch.nn as nn
import lightning as L
from minestudio.models import MinePolicy
from minestudio.train.callbacks import ObjectiveCallback
from typing import List
class MineLightning(L.LightningModule):
    
    def __init__(
        self, 
        mine_policy: MinePolicy, 
        callbacks: List[ObjectiveCallback] = [], 
        *,
        log_freq: int = 20,
        learning_rate: float = 1e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.mine_policy = mine_policy
        self.log_freq = log_freq
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps 
        self.weight_decay = weight_decay
        self.memory = None #! ???
    
    def _batch_step(self, batch, batch_idx, step_name):
        result = {'loss': 0}
        latents, self.memory = self.mine_policy(batch, self.memory)
        for callback in self.callbacks:
            call_result = callback(batch, batch_idx, step_name, latents, self.mine_policy)
            for key, val in call_result.items():
                result[key] = result.get(key, 0) + val

        if batch_idx % self.log_freq == 0:
            for key, val in result.items():
                prog_bar = ('loss' in key) and (step_name == 'train')
                self.log(f'{step_name}/{key}', val, sync_dist=False, prog_bar=prog_bar)

        return result
    
    def training_step(self, batch, batch_idx):
        return self._batch_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._batch_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.mine_policy.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/self.warmup_steps, 1)
        )
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
        }

if __name__ == '__main__':
    ...