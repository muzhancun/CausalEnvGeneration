'''
Date: 2024-11-18 20:37:50
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-18 22:32:45
FilePath: /Minestudio/minestudio/simulator/callbacks/segment.py
'''

from minestudio.simulator.utils import MinecraftGUI, GUIConstants
from minestudio.simulator.utils.gui import MaskDrawCallback
from minestudio.simulator.callbacks import MinecraftCallback

import time
from typing import Dict, Literal, Optional, Callable
from rich import print

class SegmentCallback(MinecraftCallback, MinecraftGUI):
    def __init__(
        self,
        extra_draw_call: Optional[list[Callable]] = None
    ):
        super().__init__(extra_draw_call=extra_draw_call)
        self.constants = GUIConstants()
        