'''
Date: 2024-11-10 15:52:16
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 15:30:30
FilePath: /MineStudio/minestudio/models/rocket_one/body.py
'''
import torch
import torchvision
from torch import nn
from einops import rearrange
from typing import List, Dict, Any, Tuple, Optional

import timm
from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks

class RocketPolicy(MinePolicy):
    
    def __init__(self, 
        backbone: str = 'efficientnet_b0.ra_in1k', 
        hiddim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 4,
        timesteps: int = 128,
        mem_len: int = 128,
        action_space = None,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=4)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1)
        self.pos_bias = nn.Parameter(torch.rand(1, 14 * 14, hiddim) * 0.01)
        self.pooling = nn.MultiheadAttention(hiddim, num_heads, batch_first=True) # missing positional encoding
        self.interaction = nn.Embedding(10, hiddim) # denotes the number of interaction types

        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps, 
            recurrence_type="transformer", 
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4, 
            pointwise_use_activation=False, 
            attention_mask_style="clipped_causal", 
            attention_heads=num_heads,
            attention_memory_size=mem_len + timesteps,
            n_block=num_layers,
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        b, t = input['image'].shape[:2]
        rgb = rearrange(input['image'], 'b t h w c -> (b t) c h w')
        rgb = self.transforms(rgb)

        obj_mask = input['segment']['obj_mask']
        obj_mask = rearrange(obj_mask, 'b t h w -> (b t) 1 h w')
        x = torch.cat([rgb, obj_mask], dim=1)
        x = self.backbone(x)[-1]
        x = self.updim(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        y = rearrange(input['segment']['obj_id'], 'b t -> (b t) 1')
        y = self.interaction(y + 1) # add 1 to avoid -1 index
        z = torch.cat([x, y], dim=1)
        z = z + self.pos_bias[:, :z.shape[1]] # add positional embedding
        z, _ = self.pooling(z, z, z) # (b t) p c
        z = rearrange(z.mean(dim=1), '(b t) c -> b t c', b=b, t=t)

        if not hasattr(self, 'first'):
            self.first = torch.tensor([[False]], device=z.device).repeat(b, t)
        if memory is None:
            memory = [state.to(z.device) for state in self.recurrent.initial_state(b)]
        x, memory = self.recurrent(z, self.first, memory)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_h = v_h = x
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {"pi_logits": pi_logits, "vpred": vpred}
        return latents, memory

    def initial_state(self, batch_size: int):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

def load_rocket_policy(ckpt_path: str):
    ckpt = torch.load(ckpt_path)
    model = RocketPolicy(**ckpt['hyper_parameters']['model'])
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == '__main__':
    # ckpt_path = "/nfs-shared-2/shaofei/minestudio/save/2024-11-25/14-39-15/checkpoints/step-step=120000.ckpt"
    # model = load_rocket_policy(ckpt_path).to("cuda")
    model = RocketPolicy(
        backbone='efficientnet_b2.ra_in1k', 
    ).to("cuda")
    output, memory = model(
        input={
            'image': torch.zeros(1, 8, 224, 224, 3).to("cuda"), 
            'segment': {
                'obj_id': torch.zeros(1, 8, dtype=torch.long).to("cuda"),
                'obj_mask': torch.zeros(1, 8, 224, 224).to("cuda"),
            }
        }
    )
    print(output.keys())