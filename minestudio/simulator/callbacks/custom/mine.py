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
                "start_time": 1000,
                "start_weather": "clear",
                "block": "iron_ore",
                "reward": 1.0,
                "max_reward_times": 5,
                "inventory": {"iron_ore": 1},
                "main_hand": "minecraft:iron_pickaxe",
                "random_tp_range": 100, 
            }
    """
    def __init__(self, config, openai_client=None, model=None):
        super().__init__()
        self.config = config
        self.prev_info = {}
        self.reward_memory = {}
        self.client = openai_client
        self.model = model
        self.commands = []
    
    def _fast_reset(self, sim):
        biome = random.choice(self.config['biomes'])
        x = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)
        z = np.random.randint(-self.config['random_tp_range'] // 2, self.config['random_tp_range'] // 2)
        fast_reset_commands = [
            "/effect give @s minecraft:night_vision 99999 1 true",
            "/gamemode creative",
            f"/time set {self.config['start_time']}",
            f"/weather {self.config['start_weather']}",
            f"/teleportbiome @a {biome} {x} ~0 {z}"
        ]
        for command in fast_reset_commands:
            obs, _, done, info = sim.env.execute_cmd(command)
        return False

    def _encode_image(self, image):
        # image is a numpy array
        # encode image to base64
        _, image = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(image.tobytes()).decode('utf-8')
        return image_base64

    def _get_coordinates(self, image):
        image_base64 = self._encode_image(image)
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                        <image>In this image, I want to know the coordinates of the biome {self.config[0]['biome']} from the command in left bottom corner. If the command raises an error, please return "Error", else return the coordinates of the biome in the format of "x, y, z", e.g. 100, ~, -200, where ~ means the y coordinate is not important. DO NOT OUTPUT ANYTHING ELSE.
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }]
        try:
            response = self.client.chat.completions.create(messages=message, model=self.model)
            return response.choices[0].message.content
        except Exception as e:
            return "Error"
        

    def after_reset(self, sim, obs, info):
        self.prev_info = {}
        self.reward_memory = {}

        repeat_times = 10
        while (repeat_times > 0):
            self._fast_reset(sim)
            repeat_times -= 1
            if self.config['structure'] is not None:
                obs, _, done, info = sim.env.execute_cmd(f"/locate {self.config['structure']}")

                obs, _, _, _, info = sim.step(sim.noop_action())

                # pass the info['pov'] to VLM client to extract biome coordinates
                image = info['pov']
                coordinates = self._get_coordinates(image)
                if coordinates == "Error":
                    continue
                else:
                    break

        if self.config['structure'] is not None:
            # parse coordinates
            x, y, z = coordinates.split(',')

            # teleport to the structure
            obs, _, _, info = sim.env.execute_cmd(f"/tp @s {x} {y} {z}")

        # do a attack action
        obs, _, _, info = sim.env.execute_cmd("/fill ~-1 ~-1 ~-1 ~1 ~1 ~1 air")

        # come back to survival mode
        obs, _, _, info = sim.env.execute_cmd("/gamemode survival")
        obs, _, _, _, info = sim.step(sim.noop_action())
        return obs, info
    
if __name__ == "__main__":
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        PlayCallback, FastResetCallback
    )
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            # FastResetCallback(biomes=["mountains"], random_tp_range=100),
            MineblockCallback(
                config={
                    "event": "mine_block",
                    "structure": None,
                    "start_time": 1000,
                    "start_weather": "clear",
                    "biomes": ["plains"],
                    "block": "iron_ore",
                    "reward": 1.0,
                    "max_reward_times": 5,
                    "inventory": {"iron_ore": 1},
                    "main_hand": "minecraft:iron_pickaxe",
                    "random_tp_range": 100, 
                }
            ),
            PlayCallback(),
        ]
    )

    obs, info = sim.reset()
    terminated = False
    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)
        