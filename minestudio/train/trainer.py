'''
Date: 2024-11-10 13:44:13
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:59:55
FilePath: /MineStudio/minestudio/train/trainer.py
'''
import os
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.core.mixins import HyperparametersMixin
from minestudio.models import MinePolicy
from minestudio.train.callbacks import ObjectiveCallback
from typing import List

IMPORTANT_VARIABLES = [
    "MINESTUDIO_SAVE_DIR", 
    "MINESTUDIO_DATABASE_DIR", 
]

for var in IMPORTANT_VARIABLES:
    val = os.environ.get(var, "not found")
    print(f"VAR - {var}: {val}")

def tree_detach(tree):
    if isinstance(tree, dict):
        return {k: tree_detach(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_detach(v) for v in tree]
    elif isinstance(tree, torch.Tensor):
        return tree.detach()
    else:
        return tree

class MineLightning(L.LightningModule, HyperparametersMixin):

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
        self.callbacks = callbacks
        self.log_freq = log_freq
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps 
        self.weight_decay = weight_decay
        self.memory_dict = {
            "memory": None, 
            "init_memory": None, 
            "last_timestamp": None,
        }
        self.save_hyperparameters()

    def _make_memory(self, batch):
        if self.memory_dict["init_memory"] is None:
            self.memory_dict["init_memory"] = self.mine_policy.initial_state(batch['image'].shape[0])
        if self.memory_dict["memory"] is None:
            self.memory_dict["memory"] = self.memory_dict["init_memory"]
        if self.memory_dict["last_timestamp"] is None:
            self.memory_dict["last_timestamp"] = torch.zeros(batch['image'].shape[0], dtype=torch.long).to(self.device)
        boe = batch["timestamp"][:, 0].ne(self.memory_dict["last_timestamp"] + 1)
        self.memory_dict["last_timestamp"] = batch["timestamp"][:, -1]
        # if boe's (begin-of-episode) item is True, then we keep the original memory, otherwise we reset the memory
        mem_cache = []
        for om, im in zip(self.memory_dict["memory"], self.memory_dict["init_memory"]):
            boe_f = boe[:, None, None].expand_as(om)
            mem_line = torch.where(boe_f, im, om)
            mem_cache.append(mem_line)
        self.memory_dict["memory"] = mem_cache
        return self.memory_dict["memory"]

    def _batch_step(self, batch, batch_idx, step_name):
        result = {'loss': 0}
        memory_in = self._make_memory(batch)
        latents, memory_out = self.mine_policy(batch, memory_in)
        self.memory_dict["memory"] = tree_detach(memory_out)
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