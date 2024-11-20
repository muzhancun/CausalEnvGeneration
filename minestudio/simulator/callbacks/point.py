'''
Date: 2024-11-18 20:37:50
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-18 22:32:45
FilePath: /Minestudio/minestudio/simulator/callbacks/point.py
'''

from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI, GUIConstants
from minestudio.simulator.utils.gui import PointDrawCall

import time
from typing import Dict, Literal, Optional, Callable
from rich import print
import cv2

class PointCallback(MinecraftCallback):
    def __init__(self):
        super().__init__()
        
    def after_reset(self, sim, obs, info):
        sim.callback_messages.add("Press 'P' to start pointing.")
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if info.get('P', False):
            print(f'[green]Start pointing[/green]')
        else:
            return obs, reward, terminated, truncated, info
        
        gui = MinecraftGUI(extra_draw_call=[PointDrawCall], show_info=False)
        gui.window.activate()

        while True:
            gui.window.dispatch_events()
            gui.window.switch_to()
            gui.window.set_mouse_visible(True)
            gui.window.set_exclusive_mouse(False)
            gui.window.flip()
            if gui.released_keys[gui.key.ESCAPE]:
                break
            if gui.mouse_position is not None:
                info['point'] = gui.mouse_position
            gui._show_image(info)
            
        gui.close_gui()

        if info['point'] is not None:
            print(f'[red]Stop pointing at {info["point"]}[/red]')
        return obs, reward, terminated, truncated, info
        

        
        