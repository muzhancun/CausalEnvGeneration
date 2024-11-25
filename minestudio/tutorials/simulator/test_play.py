import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, PlaySegmentCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, MaskDrawCall
from functools import partial
from minestudio.models import load_openai_policy
if __name__ == '__main__':
    agent_generator = partial(
        load_openai_policy,
        model_path="/home/zhwang/Desktop/zhancun/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/home/zhwang/Desktop/zhancun/jarvisbase/pretrained/rl-from-house-2x.weights"
    )
    sim = MinecraftSim(
        obs_size=(128, 128),
        action_type="env",
        callbacks=[
            # HumanSegmentCallback(sam_path='/home/zhwang/workspace/MineStudio/minestudio/models/realtime_sam/checkpoints', sam_choice='small'),
            PlayCallback(agent_generator=agent_generator, extra_draw_call=[RecordDrawCall, CommandModeDrawCall, MaskDrawCall]),
            RecordCallback(record_path='./output', recording=False),
        ]
    )
    obs, info = sim.reset()
    terminated = False

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()