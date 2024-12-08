from minestudio.simulator.callbacks.callback import MinecraftCallback
import random
import numpy as np
import os
import time
import threading
import subprocess

class MineflayerCallback(MinecraftCallback):
    def __init__(self, mineflayer_path, bot_name, port):
        super().__init__()
        self.mineflayer_path = mineflayer_path
        self.port = port
        self.bot_name = bot_name
    
    def after_reset(self, sim, obs, info):
        
        obs, _, _, info = sim.env.execute_cmd(f"/publish {self.port}")

        for _ in range(5):
            obs, _, _, _, info = sim.step(sim.noop_action())

        thread = threading.Thread(target=_start_mineflayer, args=(self.mineflayer_path,))
        thread.start()

        # wait for 2 seconds
        time.sleep(2)

        obs, _, _, info = sim.env.execute_cmd(f"/tp {self.bot_name} ~ ~ ~")

        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

def _start_mineflayer(mineflayer_path):
    env = os.environ.copy() 
    subprocess.run(
        ['node', mineflayer_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

if __name__ == "__main__":
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        RecordCallback, PlayCallback
    )
    import subprocess
    import threading

    mineflayer_path = "./mineflayer/index.js"
    bot_name = "Bot"
    port = 59182

    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            MineflayerCallback(mineflayer_path, bot_name, port),
            PlayCallback()
        ]

    )

    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()


       