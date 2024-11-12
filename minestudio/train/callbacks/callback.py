import torch
from typings import Dict, Any
from minestudio.models import MinePolicy
class ObjectiveCallback:
    
    def __init__(self):
        ...
    
    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy
    ) -> Dict[str, torch.Tensor]:
        return {}
