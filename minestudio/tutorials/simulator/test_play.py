import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback
)

if __name__ == '__main__':
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            PlayCallback(record_path="./output", fps=30, frame_type="pov", enable_bot=False, policy=None)
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()