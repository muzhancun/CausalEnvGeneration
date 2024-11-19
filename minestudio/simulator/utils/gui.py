'''
Date: 2024-11-15 15:15:22
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-17 22:51:25
FilePath: /Minestudio/minestudio/simulator/utils/gui.py
'''
from minestudio.simulator.utils.constants import GUIConstants   

from collections import defaultdict
from typing import List, Any, Optional, Callable
import importlib
import cv2
import time
from rich import print

def RecordDrawCall(info, **kwargs):
    if 'recording' not in info.keys():
        return info
    recording = info['recording']
    if not recording:
        return info
    arr = info['pov']
    if int(time.time()) % 2 == 0:
        cv2.circle(arr, (20, 20), 10, (255, 0, 0), -1)
        cv2.putText(arr, 'Rec', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.circle(arr, (20, 20), 10, (0, 255, 0), -1)
        cv2.putText(arr, 'Rec', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    info['pov'] = arr
    return info

def CommandModeDrawCall(info, **kwargs):
    if 'switch_command' not in info.keys():
        return info
    mode = info['switch_command']
    if not mode:
        return info
    # Draw a grey overlay on the screen
    arr = info['pov']
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    info['pov'] = arr
    return info


def MaskDrawCallback(arr, **kwargs):
    pass
    
class MinecraftGUI:
    def __init__(self, extra_draw_call: List[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.constants = GUIConstants()
        self.pyglet = importlib.import_module('pyglet')
        self.imgui = importlib.import_module('imgui')
        self.key = importlib.import_module('pyglet.window.key')
        self.mouse = importlib.import_module('pyglet.window.mouse')
        self.PygletRenderer = importlib.import_module('imgui.integrations.pyglet').PygletRenderer
        self.extra_draw_call = extra_draw_call
        self.mode = 'normal'
        self.create_window()
    
    def create_window(self):
        self.window = self.pyglet.window.Window(
            width = self.constants.WINDOW_WIDTH,
            height = self.constants.INFO_HEIGHT + self.constants.FRAME_HEIGHT,
            vsync=False,
            resizable=False
        )
        self.imgui.create_context()
        self.imgui.get_io().display_size = self.constants.WINDOW_WIDTH, self.constants.WINDOW_HEIGHT
        self.renderer = self.PygletRenderer(self.window)
        self.pressed_keys = defaultdict(lambda: False)
        self.released_keys = defaultdict(lambda: False)
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactivate = self._on_window_deactivate
        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()

        self.last_pov = None
        self.last_mouse_delta = [0, 0]
        self.capture_mouse = True
        self.chat_message = None

        self.window.clear()
        self._show_message("Waiting for reset.")

    def _on_key_press(self, symbol, modifiers):
        self.pressed_keys[symbol] = True

    def _on_key_release(self, symbol, modifiers):
        self.pressed_keys[symbol] = False
        self.released_keys[symbol] = True

    def _on_mouse_press(self, x, y, button, modifiers):
        self.pressed_keys[button] = True

    def _on_mouse_release(self, x, y, button, modifiers):
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _show_command_mode(self):
        # FIXME
        pass

    def _show_message(self, text):
        label = self.pyglet.text.Label(
            text,
            font_size=32,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x='center',
            anchor_y='center'
        )
        label.draw()
        self.window.flip()

    def _show_additional_message(self, message: List):
        if len(message) == 0:
            return
        line_height = self.constants.INFO_HEIGHT // len(message)
        y = line_height // 2
        for i, row in enumerate(message):
            line = ' | '.join(row)
            self.pyglet.text.Label(
                line,
                font_size = 7 * self.constants.SCALE, 
                x = self.window.width // 2, y = y, 
                anchor_x = 'center', anchor_y = 'center',
            ).draw()
            y += line_height

    def _update_image(self, info, message: List = [], **kwargs):
        self.window.switch_to()
        self.window.clear()
        # Based on scaled_image_display.py
        arr = info['pov']
        arr = cv2.resize(arr, dsize=(self.constants.WINDOW_WIDTH, self.constants.FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC) # type: ignore
        info['pov'] = arr
        
        if self.extra_draw_call is not None:
            for draw_call in self.extra_draw_call:
                info = draw_call(info, **kwargs)

        arr = info['pov']
        image = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, self.constants.INFO_HEIGHT)
        self._show_additional_message(message)
        
        self.imgui.new_frame()
        
        self.imgui.begin("Chat", False, self.imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed, command = self.imgui.input_text("Message", "")
        if self.imgui.button("Send"):
            self.chat_message = command
        self.imgui.end()

        self.imgui.render()
        self.renderer.render(self.imgui.get_draw_data())
        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action: dict[str, Any] = {
            name: int(self.pressed_keys[key]) for name, key in self.constants.MINERL_ACTION_TO_KEYBOARD.items()
        }

        if not self.capture_mouse:
            self.last_mouse_delta = [0, 0]
        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]
        return action
        
    def reset_gui(self):
        self.window.clear()
        self.pressed_keys = defaultdict(lambda: False)
        self._show_message("Resetting environment...")

    def _capture_mouse(self):
        release_C = self.released_keys[self.key.C]     
        if release_C:
            self.released_keys[self.key.C] = False
            self.capture_mouse = not self.capture_mouse
            self.window.set_mouse_visible(not self.capture_mouse)
            self.window.set_exclusive_mouse(self.capture_mouse)
        return release_C
    
    def _capture_control(self):
        release_L = self.released_keys[self.key.L]
        if release_L:
            self.released_keys[self.key.L] = False
        return release_L

    def _capture_recording(self):
        release_R = self.released_keys[self.key.R]
        if self.released_keys[self.key.R]:
            self.released_keys[self.key.R] = False
        return release_R
    
    def _capture_command(self):
        release_ESC = self.released_keys[self.key.ESCAPE]
        if release_ESC:
            self.released_keys[self.key.ESCAPE] = False
            print(f'[red]Command Mode Activated[/red]')
            self.mode = 'command'


    def _capture_close(self):
        # press ctrl + c to close the window
        return (self.pressed_keys[self.key.LCTRL] and self.pressed_keys[self.key.C])

    def close_gui(self):
        #! WARNING: This should be checked
        self.window.close()
        self.pyglet.app.exit()

if __name__ == "__main__":
    gui = MinecraftGUI()