import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall

if __name__ == '__main__':
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            PlayCallback(extra_draw_call=[RecordDrawCall, CommandModeDrawCall]),
            RecordCallback(record_path='./output', recording=False),
            PointCallback()
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()