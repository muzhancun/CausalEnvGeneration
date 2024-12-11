from minestudio.simulator.callbacks.callback import MinecraftCallback
import uuid
import os
import cv2
import base64, random
import numpy as np

class MineblockCallback(MinecraftCallback):
    """
        Use to generate env setting suitable for mining
        Examples:
            config = {
                "event": "mine_block",
                "structure": "village",
                "biomes": ["plains"],
                "time": 1000,
                "weather": "clear",
                "block": "iron_ore",
                "reward": 1.0,
                "inventory": {"iron_ore": 1},
                "main_hand": "minecraft:iron_pickaxe",
                "random_tp_range": 100, 
            }
    """
    def __init__(self, config, openai_client=None, model=None):
        super().__init__()
        self.config = config
        self.prev_info = {}
        self.client = openai_client
        self.model = model
    
    def _fast_reset(self, sim):
        biome = self.config['biome']
        x = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)
        z = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)

        if self.config['world'] == "nether":
            command = "/setblock ~ ~1 ~ minecraft:nether_portal"
        elif self.config['world'] == "end":
            command = "/setblock ~ ~1 ~ minecraft:end_portal"
        else:
            command = None

        fast_reset_commands = [
            "/effect give @s minecraft:night_vision 99999 1 true",
            "/effect give @s minecraft:water_breathing 99999 1 true",
            "/effect give @s minecraft:fire_resistance 99999 1 true",
            "/gamemode creative"
        ]
        if command is not None:
            fast_reset_commands.append(command)
        fast_reset_commands.extend([
            f"/time set {self.config['time']}",
            f"/weather {self.config['weather']}",
            f"/teleportbiome @a {biome} {x} ~0 {z}"
        ])
        for command in fast_reset_commands:
            obs, _, done, info = sim.env.execute_cmd(command)
        return False

    def after_reset(self, sim, obs, info):
        self.prev_info = {}

        self._fast_reset(sim)

        # do a attack action
        obs, _, _, info = sim.env.execute_cmd("/fill ~ ~ ~ ~1 ~1 ~1 air")

        # dig for {underground} blocks
        obs, _, _, info = sim.env.execute_cmd(f"/fill ~ ~ ~ ~ ~-{self.config['underground']} ~ air")

        # randomly choose facing direction from 0 to 90.0
        facing = int(random.uniform(-90.0, 90.0))
        if self.config['underground'] == 0:
            facing = int(random.uniform(-45.0, 45.0))
        obs, _, _, info = sim.env.execute_cmd(f"/tp @s ~ ~-{self.config['underground']} ~ -90 {facing}")

        prev_y = None
        repeat_times = 20
        while (repeat_times > 0 and self.config['underground'] < 0):
            repeat_times -= 1
            for _ in range(5):
                obs, _, _, _, info = sim.step(sim.noop_action())
            # import ipdb; ipdb.set_trace()
            if prev_y is not None and abs(info['player_pos']['y'] - prev_y) < 1:
                break
            prev_y = info['player_pos']['y']
        obs, _, _, info = sim.env.execute_cmd(f"/tp @s ~ ~ ~ -90 {facing}")
        if self.config['generate_block']:
            # place the target block
            if  0 <= facing < 45:
                # randomly choose the y coordinate
                y = 1
                obs, _, _, info = sim.env.execute_cmd(f"/setblock ~1 ~{y} ~ {self.config['block']}")
            elif facing >= 45:
                # randomly choose y to be -1 or 0
                y = random.choice([-1, 0])
                if self.config['pos'] != y:
                    y = self.config['pos']
                if y == -1:
                    obs, _, _, info = sim.env.execute_cmd(f"/setblock ~ ~{y} ~ {self.config['block']}")
                else:
                    obs, _, _, info = sim.env.execute_cmd(f"/setblock ~1 ~{y} ~ {self.config['block']}")
            elif -45 <= facing < 0:
                # randomly choose y coordinate from 0 to 2
                y = 1
                obs, _, _, info = sim.env.execute_cmd(f"/setblock ~1 ~{y} ~ {self.config['block']}")
            elif facing < -45:
                # randomly choose y coordinate from 2 to 3
                y = random.choice([2, 3])
                obs, _, _, info = sim.env.execute_cmd(f"/setblock ~1 ~{y} ~ {self.config['block']}")

        # come back to survival mode
        obs, _, _, info = sim.env.execute_cmd("/gamemode survival")

        for slot, item in self.config['armor'].items():
            if item == "":
                continue
            obs, _, _, info = sim.env.execute_cmd(f"/replaceitem entity @s {slot} {item}")

        for block, num in self.config['inventory'].items():
            obs, _, _, info = sim.env.execute_cmd(f"/give @s {block} {num}")

        obs, _, _, _, info = sim.step(sim.noop_action())
        self.prev_info = info.copy()
        # obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        override_reward = 0.
        event_type = self.config['event']
        block = self.config['block']
        delta = self._get_obj_num(info, event_type, block) - self._get_obj_num(self.prev_info, event_type, block)
        self.prev_info = info.copy()
        if delta <= 0:
            return obs, reward, terminated, truncated, info
        else:
            override_reward = self.config['reward']
            terminated = True
            return obs, override_reward, terminated, truncated, info
        
    def _get_obj_num(self, info, event_type, obj):
        if event_type not in info:
            return 0.
        if obj not in info[event_type].keys():
            return 0.
        res = info[event_type][obj]
        return res.item() if isinstance(res, np.ndarray) else res 

if __name__ == "__main__":
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        PlayCallback, FastResetCallback
    )
    from openai import OpenAI
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # openai_api_base = os.getenv("OPENAI_API_BASE")
    # client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            MineblockCallback(
                config={
                    "time": 13000,
                    "weather": "clear",
                    "biome": "taiga",
                    "world": "overworld",
                    "block": "spruce_sign",
                    "inventory": {"minecraft:wooden_axe": 1, "minecraft:torch": 10, "minecraft:bread": 5},
                    "underground": 0,
                    "armor": {"armor.head": "", "armor.chest": "", "armor.legs": "", "armor.feet": "", "weapon.mainhand": "minecraft:wooden_axe", "weapon.offhand": ""},
                    "reward": 1.0,
                    "random_tp_range": 100,
                    "generate_block": True,
                    "event": "mine_block",
                    "in_air": False,
                    # "underground": 30, # number of blocks to dig
                    # "world": "nether",
                    # "structure": None,
                    # "time": 1000,
                    # "weather": "clear",
                    # "biomes": ["soul_sand_valley"],
                    # "block": "iron_ore",
                    # "reward": 1.0,
                    # "max_reward_times": 5,
                    # "inventory": {"minecraft:dirt": 4, "minecraft:torch": 16, "minecraft:crafting_table": 1},
                    # "main_hand": "minecraft:iron_pickaxe",
                    # "random_tp_range": 100, 
                }
            ),
            PlayCallback(),
        ],
    )

    obs, info = sim.reset()
    terminated = False
    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)