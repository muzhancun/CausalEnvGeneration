'''
Date: 2024-11-15 15:15:22
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-16 02:10:22
FilePath: /minestudio/simulator/utils/gui.py
'''
from minestudio.simulator.utils.constants import *

import pyglet
import imgui
from imgui.integrations.pyglet import PygletRenderer
from collections import defaultdict
from typing import List, Any

class MinecraftGUI:
    def __init__(self):
        self.create_window()
    
    def create_window(self):
        self.window = pyglet.window.Window(
            width=WINDOW_WIDTH,
            height=INFO_HEIGHT + FRAME_HEIGHT,
            vsync=False,
            resizable=False
        )
        imgui.create_context()
        imgui.get_io().display_size = WINDOW_WIDTH, WINDOW_HEIGHT
        self.renderer = PygletRenderer(self.window)
        self.pressed_keys = defaultdict(lambda: False)
        self.released_keys = defaultdict(lambda: False)
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactive = self._on_window_deactivate
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
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER
        
    def _show_message(self, text):
        label = pyglet.text.Label(
            text,
            font_size=32,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x='center',
            anchor_y='center'
        )
        label.draw()
        self.window.flip()

    def _update_image(self, arr, message: List = [], **kwargs):
        self.window.switch_to()
        self.window.clear()
        # Based on scaled_image_display.py
        arr = cv2.resize(arr, dsize=(WINDOW_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC) # type: ignore
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, INFO_HEIGHT)
        self._show_additional_message(message)
        
        imgui.new_frame()

        if self.extra_draw_call:
            self.extra_draw_call()
        
        imgui.begin("Chat", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed, command = imgui.input_text("Message", "")
        if imgui.button("Send"):
            self.chat_message = command
        imgui.end()

        imgui.render()
        self.renderer.render(imgui.get_draw_data())
        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action: dict[str, Any] = {
            name: int(self.pressed_keys[key]) for name, key in MINERL_ACTION_TO_KEYBOARD.items()
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

    def capture_mouse(self):
        release_C = self.released_keys[key.C]     
        if release_C:
            self.released_keys[key.C] = False
            self.capture_mouse = not self.capture_mouse
            self.window.set_mouse_visible(not self.capture_mouse)
            self.window.set_exclusive_mouse(self.capture_mouse)
        return release_C
    
    def capture_control(self):
        release_L = self.released_keys[key.L]
        if release_L:
            self.released_keys[key.L] = False
        return release_L

    def capture_recording(self):
        release_R = self.released_keys[key.R]
        if self.released_keys[key.R]:
            self.released_keys[key.R] = False
        return release_R
            
    def close_gui(self):
        #! WARNING: This should be checked
        self.window.close()
        pyglet.app.exit()