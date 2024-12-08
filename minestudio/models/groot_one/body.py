'''
Date: 2024-11-25 07:03:41
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 12:51:34
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


class LatentSpace(nn.Module):

    def __init__(self, hiddim: int) -> None:
        super().__init__()
        self.encode_mu = nn.Linear(hiddim, hiddim)
        self.encode_log_var = nn.Linear(hiddim, hiddim)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.encode_mu(x)
        log_var = self.encode_log_var(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return { 'mu': mu, 'log_var': log_var, 'z': z }

class VideoEncoder(nn.Module):
    
    def __init__(self, hiddim: int, num_layers: int = 8, num_heads: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.MultiheadAttention(hiddim, num_heads, batch_first=True) # missing positional encoding
        self.pos_bias = nn.Parameter(torch.rand(1, 14 * 14, hiddim) * 0.01)
        self.encode_video = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = hiddim,
                nhead = num_heads,
                dim_feedforward = hiddim*2,
                dropout = dropout,
            ),
            num_layers = num_layers
        )
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, images: torch.Tensor) -> Dict:
        """
        images: (b, t, c, h, w)
        """
        x = rearrange(images, 'b t c h w -> (b t) (h w) c')
        x = x + self.pos_bias[:, :x.shape[1]]
        x, _ = self.pooling(x, x, x)
        x = rearrange(x.mean(dim=1), '(b t) c -> b t c', b=images.shape[0])
        x = self.encode_video(x)
        x = x.mean(dim=1) # b c
        dist = self.encode_dist(x)
        return dist


class ImageEncoder(nn.Module):
    
    def __init__(self, hiddim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.MultiheadAttention(hiddim, num_heads, batch_first=True) # missing positional encoding
        self.pos_bias = nn.Parameter(torch.rand(1, 14 * 14, hiddim) * 0.01)
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, image: torch.Tensor) -> Dict:
        """
        image: (b, c, h, w)
        """
        x = rearrange(image, 'b c h w -> b (h w) c')
        x = x + self.pos_bias[:, :x.shape[1]]
        x, _ = self.pooling(x, x, x)
        x = x.mean(dim = 1)
        dist = self.encode_dist(x)
        return dist


class Decoder(nn.Module):
    
    def __init__(
        self, 
        hiddim: int, 
        num_heads: int = 16,
        num_layers: int = 4, 
        timesteps: int = 128, 
        mem_len: int = 128, 
    ) -> None:
        super().__init__()
        self.hiddim = hiddim
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

    def forward(self, x: torch.Tensor, memory: List) -> Tuple[torch.Tensor, List]:
        b, t = x.shape[:2]
        if not hasattr(self, 'first'):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        x, memory = self.recurrent(x, self.first, memory)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        return x, memory

    def initial_state(self, batch_size: int):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

class GrootPolicy(MinePolicy):
    
    def __init__(
        self, 
        backbone: str = 'efficientnet_b0.ra_in1k', 
        freeze_backbone: bool = True,
        hiddim: int = 1024,
        encoder_kwarg: Dict = {}, 
        decoder_kwarg: Dict = {},
        action_space=None,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1)
        self.video_encoder = VideoEncoder(hiddim, **encoder_kwarg)
        self.image_encoder = ImageEncoder(hiddim, **encoder_kwarg)
        self.decoder = Decoder(hiddim, **decoder_kwarg)
        self.fuse = nn.MultiheadAttention(hiddim, num_heads, batch_first=True) # missing positional encoding
        self.pos_bias = nn.Parameter(torch.rand(1, 14 * 14, hiddim) * 0.01)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        b, t = input['image'].shape[:2]

        image = rearrange(input['image'], 'b t h w c -> (b t) c h w')
        image = self.transforms(image)
        image = self.backbone(image)[-1]
        image = self.updim(image)
        image = rearrange(image, '(b t) c h w -> b t c h w', b=b)

        if 'reference' in input:
            reference = input['reference']
            reference = rearrange(reference, 'b t h w c -> (b t) c h w')
            reference = self.transforms(reference)
            reference = self.backbone(reference)[-1]
            reference = self.updim(reference)
            reference = rearrange(reference, '(b t) c h w -> b t c h w', b=b)
        else:
            reference = image

        posterior_dist = self.video_encoder(reference)
        prior_dist = self.image_encoder(reference[:, 0])

        z = dist['z'].unsqueeze(1)
        x = rearrange(image, 'b t c h w -> (b t) (h w) c')
        x = torch.cat([x, z], dim=1)
        x = x + self.pos_bias[:, :x.shape[1]]
        x, _ = self.fuse(x, x, x)

        x = rearrange(x.mean(dim=1), '(b t) c -> b t c', b=b)
        x, memory = self.decoder(x, memory)
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
            "posterior_dist": posterior_dist, 
            "prior_dist": prior_dist
        }
        return latents, memory

    def initial_state(self, batch_size: int):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

if __name__ == '__main__':
    model = GrootPolicy(
        backbone = 'efficientnet_b0.ra_in1k'
    ).to("cuda")
    memory = None
    output, memory = model(input, memory)