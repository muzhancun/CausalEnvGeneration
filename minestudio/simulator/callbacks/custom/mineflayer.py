from minestudio.simulator.callbacks.callback import MinecraftCallback
import random
import numpy as np
import os

class MineflayerCallback(MinecraftCallback):
    def __init__(self, mineflayer_path, bot_name, port):
        super().__init__()
        self.mineflayer_path = mineflayer_path
        self.port = port
        self.bot_name = bot_name
    
    def after_reset(self, sim, obs, info):
        
        obs, _, _, info = sim.env.execute_cmd(f"/publish {self.port}")      

        return obs, info

if __name__ == "__main__":
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        RecordCallback
    )
    mineflayer_path = "./mineflayer"
    bot_name = "Bot"
    port = 59181

    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            MineflayerCallback(mineflayer_path, bot_name, port),
            # RecordCallback(record_path="./record", frame_type='pov', recording=True)
        ]

    )

    obs, info = sim.reset()
    terminated = False

    try:
        os.subprocess.run(f'cd {mineflayer_path} && DEBUG="minecraft-protocol" node index.js')

        # tp the bot
        obs, _, _, info = sim.env.execute_cmd(f"/tp {bot_name} ~ ~ ~")
    except:
        print("Error running Mineflayer bot")  

    for i in range(100):
        obs, reward, terminated, truncated, info = sim.step(sim.noop_action())

    sim.close()


       