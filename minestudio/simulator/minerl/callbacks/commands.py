'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-11 19:33:46
FilePath: /MineStudio/minestudio/simulator/minerl/callbacks/commands.py
'''

from minestudio.simulator.minerl.callbacks.callback import MinecraftCallback

class CommandsCallback(MinecraftCallback):
    
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
    
    def after_reset(self, sim, obs, info):
        for command in self.commands:
            obs, reward, done, info = sim.env.execute_cmd(command)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info