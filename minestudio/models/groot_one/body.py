'''
Date: 2024-11-25 07:03:41
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 07:12:12
FilePath: /MineStudio/minestudio/models/groot_one/body.py
'''
import torch
import torchvision
from torch import nn
from einops import rearrange
from typing import List, Dict, Any, Tuple, Optional

import timm
from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        pass

class GrootPolicy(MinePolicy):
    
    def __init__(
        backbone: str = 'efficientnet_b0.ra_in1k', 
        hiddim: int = 1024,
        encoder_kwarg: Dict = {}, 
        decoder_kwarg: Dict = {},
        action_space=None,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)


    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        pass

    def initial_state(self, batch_size: int):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

if __name__ == '__main__':
    model = GrootPolicy(
        backbone = 'efficientnet_b0.ra_in1k'
    ).to("cuda")
    
    output, memory = model(input, memory)