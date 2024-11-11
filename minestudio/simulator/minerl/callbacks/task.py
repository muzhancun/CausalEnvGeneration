'''
Date: 2024-11-11 19:29:45
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-11 19:45:13
FilePath: /MineStudio/minestudio/simulator/minerl/callbacks/task.py
'''
import random
from minestudio.simulator.minerl.callbacks.callback import MinecraftCallback

class TaskCallback(MinecraftCallback):
    
    def __init__(self, task_cfg):
        """
        TaskCallback 
        Example:
            task_cfg = [{
                'name': 'chop tree',
                'text': 'chop the tree', 
            }]
        """
        super().__init__()
        self.task_cfg = task_cfg
    
    def after_reset(self, sim, obs, info):
        task = random.choice(self.task_cfg)
        print(f"Switching to task: {task['name']}.")
        obs["task"] = task
        return obs, info