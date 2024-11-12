'''
Date: 2024-11-12 13:59:08
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 17:01:34
FilePath: /MineStudio/minestudio/train/callbacks/behavior_clone.py
'''

import torch
from typings import Dict, Any
from minestudio.models import MinePolicy
from minestudio.train.callbacks.callback import ObjectiveCallback

class BehaviorCloneCallback(ObjectiveCallback):
        
    def __init__(self):
        super().__init__()

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy
    ) -> Dict[str, torch.Tensor]:
        return {}