'''
Date: 2024-11-11 05:20:17
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-11 07:55:26
FilePath: /MineStudio/minestudio/simulator/minerl/entry.py
'''

import os
import cv2
import argparse
import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal

from minestudio.utils.vpt_lib.actions import ActionTransformer
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.simulator.minerl.utils.inventory import map_slot_number_to_cmd_slot
from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minestudio.simulator.minerl.callbacks import MinecraftCallback

if not os.path.exists(os.path.join(os.path.dirname(__file__), "MCP-Reborn")):
    print("Detecting missing MCP-Reborn, downloading...")
    import huggingface_hub, zipfile
    huggingface_hub.hf_hub_download(repo_id='phython96/ROCKET-MCP-Reborn', filename='MCP-Reborn.zip', local_dir='.')
    with zipfile.ZipFile('MCP-Reborn.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove('MCP-Reborn.zip')

class MinecraftSim(gymnasium.Env):
    
    def __init__(
        self, 
        obs_size: Tuple[int, int] = (224, 224),         # the resolution of the observation (cv2 resize)
        action_type: Literal['env', 'agent'] = 'agent', # the style of the action space
        render_size: Tuple[int, int] = (640, 360),      # the original resolution of the game is 640x360
        seed: int = 0,                                  # the seed of the minecraft world
        inventory: Dict = {},                           # the initial inventory of the agent
        preferred_spawn_biome: Optional[str] = None,    # the preferred spawn biome when call reset 
        num_empty_frames: int = 20,                     # the number of empty frames to skip when calling reset
        callbacks: List[MinecraftCallback] = [],        # the callbacks to be called before and after each basic calling
        **kwargs
    ) -> Any:
        super().__init__()
        self.obs_size = obs_size
        self.action_type = action_type
        self.render_size = render_size
        self.seed = seed
        self.num_empty_frames = num_empty_frames
        self.callbacks = callbacks
        
        self.env = HumanSurvival(
            fov_range = [70, 70],
            gamma_range = [2, 2],
            guiscale_range = [1, 1],
            cursor_size_range=[16.0, 16.0],
            frameskip = 1,
            resolution = render_size, 
            inventory = inventory,
            preferred_spawn_biome = preferred_spawn_biome, 
        ).make()
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        for callback in self.callbacks:
            action = callback.before_step(self, action)
        obs, reward, done, info = self.env.step(action.copy()) 
        terminated, truncated = done, done
        obs, info = self._wrap_obs_info(obs, info)
        for callback in self.callbacks:
            obs, reward, terminated, truncated, info = callback.after_step(self, obs, reward, terminated, truncated, info)
        self.obs, self.info = obs, info
        return obs, reward, terminated, truncated, info

    def reset(self) -> Tuple[np.ndarray, Dict]:
        reset_flag = True
        for callback in self.callbacks:
            reset_flag = callback.before_reset(self, reset_flag)
        if reset_flag: # hard reset
           self.env.reset()
        for _ in range(self.num_empty_frames): # skip the frames to avoid the initial black screen
            action = self.env.action_space.no_op()
            obs, reward, done, info = self.env.step(action)
        obs, info = self._wrap_obs_info(obs, info)
        for callback in self.callbacks:
            obs, info = callback.after_reset(self, obs, info)
        self.obs, self.info = obs, info
        return obs, info

    def _wrap_obs_info(self, obs: Dict, info: Dict) -> Dict:
        _info = info.copy()
        _info.update(obs)
        _obs = {'image': cv2.resize(obs['pov'], dsize=self.obs_size, interpolation=cv2.INTER_LINEAR)}
        return _obs, _info

    def noop_action(self) -> Dict[str, Any]:
        if self.action_type == 'agent':
            return {
                "buttons": [0],
                "camera": [60],
            }
        else:
            return self.env.action_space.no_op()

    def close(self) -> None:
        for callback in self.callbacks:
            callback.before_close(self)
        close_status = self.env.close()
        for callback in self.callbacks:
            callback.after_close(self)
        return close_status

    def render(self) -> None:
        for callback in self.callbacks:
            callback.before_render(self)
        #! core logic
        for callback in self.callbacks:
            callback.after_render(self)

    @property
    def action_space(self) -> spaces.Dict:
        if self.action_type == 'agent':
            return gymnasium.spaces.Dict({
                "buttons": gymnasium.spaces.MultiDiscrete([8641]),
                "camera":  gymnasium.spaces.MultiDiscrete([121]), 
            })
        elif self.action_type == 'env':
            # using factored action space
            ...
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    @property
    def observation_space(self) -> spaces.Dict:
        width, height = self.obs_size
        return gymnasium.spaces.Dict({
            "image": gymnasium.spaces.Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8)
        })

if __name__ == '__main__':
    from minestudio.simulator.minerl.callbacks import SpeedTestCallback
    sim = MinecraftSim(
        callbacks=[SpeedTestCallback(50)]
    )
    obs, info = sim.reset()
    action = sim.noop_action()
    for i in range(200):
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()