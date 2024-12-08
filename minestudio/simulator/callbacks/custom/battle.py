from minestudio.simulator.callbacks.callback import MinecraftCallback
import random
import numpy as np

class BattleCallback(MinecraftCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def _fast_reset(self, sim):
        biome = random.choice(self.config['biomes'])
        x = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)
        z = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)
        fast_reset_commands = [
            "/effect give @s minecraft:night_vision 99999 1 true",
            "/effect give @s minecraft:water_breathing 99999 1 true",
            "/effect give @s minecraft:fire_resistance 99999 1 true",
            "/gamemode creative",
            f"/time set {self.config['start_time']}",
            f"/weather {self.config['start_weather']}",
            f"/teleportbiome @a {biome} {x} ~0 {z}"
        ]
        for command in fast_reset_commands:
            obs, _, done, info = sim.env.execute_cmd(command)
        return False
    
    def after_reset(self, sim, obs, info):
        self.prev_info = {}
        
        self._fast_reset(sim)
        
        obs, _, _, info = sim.env.execute_cmd(f"/execute in minecraft:{self.config['world']} run teleport @s ~ ~ ~")

        obs, _, _, info = sim.env.execute_cmd("/fill ~ ~ ~ ~ ~1 ~ air")
        obs, _, _, info = sim.env.execute_cmd(f"/gamemode survival")

        for slot, item in self.config['armor'].items():
            obs, _, _, info = sim.env.execute_cmd(f"/replaceitem entity @s {slot} {item}")

        for block, num in self.config['inventory'].items():
            obs, _, _, info = sim.env.execute_cmd(f"/give @s {block} {num}")

        # summon mobs
        for mob in self.config['mobs']:
            for _ in range(mob['number']):
                name = mob['name']
                x = sim.np_random.uniform(*mob['range_x'])
                z = 0
                chat = f'/execute as @p at @p run summon minecraft:{name} ~{x} ~ ~{z} {{Age:0}}'
                obs, _, _, info = sim.env.execute_cmd(chat)

        self.prev_info = info.copy()
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        override_reward = 0.
        event_type = self.config['event']
        entity = self.config['entity']
        delta = self._get_obj_num(info, event_type, entity) - self._get_obj_num(self.prev_info, event_type, entity)
        self.prev_info = info.copy()
        if delta <= 0:
            return obs, override_reward, terminated, truncated, info
        else:
            override_reward += self.config['reward']
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
        PlayCallback
    )

    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            # FastResetCallback(biomes=["mountains"], random_tp_range=100),
            BattleCallback(
                config={
                    "event": "kill_entity",
                    "structure": None,
                    "start_time": "night",
                    "start_weather": "clear",
                    "world": "the_end",
                    "biomes": ["mountains"],
                    "entity": "ender_dragon",
                    "mobs": [
                        {'name': 'ender_dragon', 'number': 1, 'range_x': [0, 100]}
                    ],
                    "reward": 1.0,
                    "inventory": {"minecraft:dirt": 4, "minecraft:torch": 16, "minecraft:crafting_table": 1},
                    "armor": {
                        "armor.head": "minecraft:diamond_helmet",
                        "armor.chest": "minecraft:diamond_chestplate",
                        "armor.legs": "minecraft:diamond_leggings",
                        "armor.feet": "minecraft:diamond_boots",
                        "weapon.mainhand": "minecraft:diamond_sword",
                    },
                    "random_tp_range": 100, 
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
        print(reward)
