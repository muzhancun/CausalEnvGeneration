import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, HumanSegmentCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, MaskDrawCall

if __name__ == '__main__':
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            HumanSegmentCallback(sam_path='/home/zhwang/workspace/MineStudio/minestudio/models/realtime_sam/checkpoints', sam_choice='small'),
            PlayCallback(extra_draw_call=[RecordDrawCall, CommandModeDrawCall, MaskDrawCall]),
            RecordCallback(record_path='./output', recording=False),
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()