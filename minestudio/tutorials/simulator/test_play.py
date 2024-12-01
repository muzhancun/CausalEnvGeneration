import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, PlaySegmentCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, SegmentDrawCall
from functools import partial
from minestudio.models import load_openai_policy, load_rocket_policy
if __name__ == '__main__':
    agent_generator = partial(
        load_rocket_policy,
        ckpt_path="/home/zhwang/Desktop/zhancun/jarvisbase/pretrained/rocket_12-01.ckpt",
    )
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            PlaySegmentCallback(sam_path='/home/zhwang/workspace/MineStudio/minestudio/models/realtime_sam/checkpoints', sam_choice='small'),
            PlayCallback(agent_generator=agent_generator, extra_draw_call=[RecordDrawCall, CommandModeDrawCall, SegmentDrawCall]),
            RecordCallback(record_path='./output', recording=False),
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()