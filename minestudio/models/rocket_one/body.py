'''
Date: 2024-11-10 15:52:16
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 16:10:53
FilePath: /MineStudio/minestudio/models/rocket_one/body.py
'''
import torch
import torchvision
from torch import nn
from einops import rearrange
from typing import List, Dict, Any, Tuple, Optional

import timm
from transformers import TransfoXLConfig, TransfoXLModel

class RocketOnePolicy(nn.Module):
    
    def __init__(self, 
        backbone: str = 'efficientnet_b0.ra_in1k', 
        hiddim: int = 1024, 
        num_heads: int = 16,
        num_layers: int = 12,
        mem_len: int = 128,
    ):
        super(RocketOnePolicy, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=4)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1)
        self.pooling = nn.MultiheadAttention(hiddim, num_heads, batch_first=True)
        
        transfoXL = TransfoXLConfig(
            d_model=hiddim,
            d_embed=hiddim,
            n_head=num_heads,
            d_head=hiddim // num_heads,
            n_layer=num_layers,
            mem_len=mem_len,
        )
        self.recurrent = TransfoXLModel(transfoXL)


    def forward(self, input: Dict, memory: Optional[List[torch.FloatTensor]] = None) -> Dict:
        b, t = input['image'].shape[:2]
        rgb = rearrange(input['image'], 'b t c h w -> (b t) c h w')
        rgb = self.transforms(rgb)
        
        obj_mask = input['segment']['obj_mask']
        obj_mask = rearrange(obj_mask, 'b t h w -> (b t) 1 h w')
        x = torch.cat([rgb, obj_mask], dim=1)
        x = self.backbone(x)[-1]
        x = self.updim(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, _ = self.pooling(x, x, x)
        x = rearrange(x.mean(dim=1), '(b t) c -> b t c', b=b, t=t)
        # import ipdb; ipdb.set_trace()
        output = self.recurrent(inputs_embeds=x, mems=memory, return_dict=True)
        memory = output['mems']
        last_hidden_state = output['last_hidden_state']
        
        return last_hidden_state, memory

if __name__ == '__main__':
    model = RocketOnePolicy(
        backbone='efficientnet_b0.ra_in1k', 
    ).to("cuda")
    output, memory = model(
        input={
            'image': torch.zeros(1, 8, 3, 224, 224).to("cuda"), 
            'segment': {
                'obj_id': torch.zeros(1, 8).to("cuda"),
                'obj_mask': torch.zeros(1, 8, 224, 224).to("cuda"),
            }
        }
    )
    print(output.shape)
    # import ipdb; ipdb.set_trace()