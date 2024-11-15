'''
Date: 2024-11-14 20:10:54
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-16 02:06:38
FilePath: /minestudio/simulator/callbacks/play.py
'''
from minestudio.simulator.callbacks import RecordCallback
from minestudio.simulator.utils.constants import *
from minestudio.simulator.utils import MinecraftGUI

import time
from typing import Dict, Literal, Optional
from rich import print

class PlayCallback(RecordCallback, MinecraftGUI):
    def __init__(
        self, 
        record_path: str, 
        fps: int = 20, 
        frame_type: Literal['pov', 'obs'] = 'pov',
        enable_bot: bool = False,
        policy: Optional[Dict] = None,
    ):
        super().__init__(record_path, fps, frame_type, recording=False)
        self.start_time = time.time()
        self.end_time = time.time()
        self.enable_bot = enable_bot
        self.switch = 'human'
        self.terminated = False
        self.last_obs = None
        self.last_action = None
        self.timestep = 0

        if enable_bot:
            assert policy is not None, 'Policy must be specified if enable_bot is True'
            self.policy = load_policy(policy) #! TODO: load policy with a dict of name and weights
            self.reset_policy()

        # print a table of key bindings
        print(
            f'[yellow]Extra Key Bindings Besides Minecraft Controls: \n'
            f'  [white]C[/white]: Capture Mouse \n'
            f'  [white]L[/white]: Switch Control \n'
            f'  [white]R[/white]: Start/Stop Recording \n'
        )
    
    def reset_policy(self):
        self.memory = self.policy.initial_state()

    def before_reset(self, sim, reset_flag):
        super().before_reset(sim, reset_flag)
        self.reset_gui()
        self.terminated = False
        if self.enable_bot:
            self.reset_policy()
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        if self.frame_type == 'obs':
            self._update_image(obs['image'])
        elif self.frame_type == 'pov':
            self._update_image(info['pov'])
        else:
            raise ValueError(f'Invalid frame_type: {self.frame_type}')
        self.last_obs = obs
        self.timestep = 0
        return obs, info
    
    def before_step(self, sim, action):
        """
        Step environment for one frame.

        If `action` is not str or None, assume it is a valid action and pass it to the environment.
        If `action` is `human`, read action from player (current keyboard/mouse state).
        Else, assume it is a policy and execute the action from the policy.

        The executed action will be added to the info dict as "taken_action".
        """
        assert not self.terminated, "Cannot step environment after it is done."

        self.window.dispatch_events()
        if isinstance(action, str) or action is None:
            if action == 'human':
                human_action = self._get_human_action()
                action = human_action
            else:
                assert self.enable_bot, "Policy is not specified."
                policy_action = self.policy.get_action(self.last_obs, self.memory, input_shape = "*")
                policy_action = sim.agent_action_to_env_action(policy_action)
                action = policy_action
        
        if self.chat_message is not None:
            action["chat"] = self.chat_message
            self.chat_message = None

        self.last_action = action
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        super().after_step(sim, obs, reward, terminated, truncated, info)
        self.terminated = terminated
        self.timestep += 1
        self.last_obs = obs
        self.end_time = time.time()
        time.sleep(max(0, MINERL_FRAME_TIME - (self.end_time - self.start_time)))
        fps = 1 / (self.end_time - self.start_time)
        self.start_time = time.time()
        message = [
            [f"Role: {self.switch}"], 
            [f"Record: {self.recording}", f"Record Step: {len(self.frames)}", f"Timestep: {self.timestep}", f"FPS: {fps:.2f}"], 
            [f"X: {info['player_pos']['x']:.2f}", f"Y: {info['player_pos']['y']:.2f}", f"Z: {info['player_pos']['z']:.2f}"],
        ]
        action_items = []
        for k, v in self.last_action.items():
            if k == 'camera':
                v = f"({v[0]:.2f}, {v[1]:.2f})"
            elif 'hotbar' in k:
                continue
            action_items.append(f"{k}: {v}")
        message.append(action_items)
        self._update_image(info["pov"], message=message)

        # press 'C' to capture mouse
        release_C = self.capture_mouse()

        # press 'L' to switch control
        switch_control = self.capture_control()
        if switch_control:
            self.switch = 'human' if self.switch == 'bot' else 'bot'
            print(f'[red]Switch to {self.switch} control[/red]')

        # press 'R' to start/stop recording\
        switch_recording = self.capture_recording()
        if switch_recording:
            if self.recording:
                print(f'[red]Stop recording[/red]')
                self._save_episode()
            else:
                print(f'[green]Start recording[/green]')
            self.recording = not self.recording
        
        if terminated:
            self._show_message("Episode terminated.")

        info["taken_action"] = self.last_action
        info['switch'] = self.switch
        self.terminated = terminated

        if switch_control:
            #? TODO: add more features after switching control
            if self.switch == 'bot':
                if not self.enable_bot:
                    print('[red]Policy is not specified, switch to human control[/red]')
                    self.switch = 'human'
                else:
                    self.reset_policy()
            obs, reward, terminated, truncated, info = sim.step(sim.noop_action())

        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        super().before_close(sim)
        self.close_gui()

        