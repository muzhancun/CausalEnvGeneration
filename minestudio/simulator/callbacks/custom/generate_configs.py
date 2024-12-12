'''
Date: 2024-12-12 00:06:47
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-12-12 02:03:15
FilePath: /MineStudio/minestudio/simulator/callbacks/custom/generate_configs.py
'''
import json
# read respones.jsonl

responses = []
with open('responses.jsonl') as f:
    for line in f:
        responses.append(json.loads(line))

# read mine.jsonl

mines = []
with open('mine.jsonl') as f:
    for line in f:
        mines.append(json.loads(line))

import random
import numpy as np
import minecraft_data
mcd = minecraft_data("1.16.5")
item_list = mcd.items_list
block_list = mcd.blocks_list
# merge item_list and block_list
item_list.extend(block_list)
item_list = [item['name'] for item in item_list]


from tqdm import tqdm
import openai
import requests
import os

openai_key = "sk-o2KHVcwHYwI6E8X03e883bD19bB74f3cA25fA65c3fBe83A3"
url = "https://www.dwyu.top/v1/chat/completions"
client = openai.Client(api_key=openai_key, base_url="https://www.dwyu.top/v1/chat/completions")

for mine in tqdm(mines):
    # change the space to underscore
    name = mine.replace(' ', '_')

    # find response's title == mine

    response = None
    for res in responses:
        if res['title'] == mine:
            response = res
            break

    ret = []
    for item in item_list:
        if name in item:
            ret.append(item)
    if len(ret) == 0:
        continue
    mine = random.choice(ret)

    # write prompt
    prompt = "I need you to generate environment configs for reinforcement learning in Minecraft.\n"
    prompt += "The task is to mine a certain block.\n"
    prompt += f"Now give you the object {name}.\n"
    prompt += f"First determine whether the object {name} itself is suitable for mining and is a specific block in Minecraft.\n"
    prompt += "For example, if the object is `ore`, it is not suitable for mining.\n"
    prompt += "If it is not suitable for mining, please output 'Not suitable for mining', do not output anything else.\n"
    prompt += "Then, if the object is suitable for mining, please output the environment configs for mining the object according to the relationships and properties of the object.\n"
    prompt += "Here is an example of relationship and properties of item iron ore:\n"
    prompt += f'"{response["relationships"]}", "{response["properties"]}"\n'
    prompt += "Now please output the environment configs for mining the object {mine} in json format.\n"
    prompt += "You should include the following keys, for example (of iron ore):\n"
    prompt += "```\n"
    prompt += "{\n"
    prompt += '    "time": 1000, # the tme of Minecraft, 1000 is morning, 13000 is night beginning, 18000 is midnight, 23000 is night ending,\n'
    prompt += '    "weather": "clear", # the weather of Minecraft, "clear", "rain", "thunder,\n'
    prompt += '    "biome": "plains", # the biome of Minecraft in overworld: [\'badlands\', \'bamboo_jungle\', \'beach\', \'birch_forest\', \'cold_ocean\', \'deep_frozen_ocean\', \'deep_lukewarm_ocean\', \'deep_ocean\', \'desert\', \'flower_forest\', \'forest\', \'frozen_ocean\', \'frozen_river\', \'ice_spikes\', \'jungle\', \'lukewarm_ocean\', \'mushroom_fields\', \'ocean\', \'plains\', \'river\', \'savanna\', \'savanna_plateau\', \'snowy_beach\', \'snowy_taiga\', \'sunflower_plains\', \'swamp\', \'taiga\', \'warm_ocean\'], in nether: [\'basalt_deltas\', \'crimson_forest\', \'nether_wastes\', \'soul_sand_valley\', \'warped_forest\'], in end: [\'end_barrens\', \'end_highlands\', \'end_midlands\', \'small_end_islands\'], **You should only choose one biome for the object from the list above**,\n'
    prompt += '    "world": "overworld", # the world of Minecraft, "overworld", "nether", "end",\n'
    prompt += '    "block": "iron_ore", # the block to be mined,\n'
    prompt += '    "inventory": {"minecraft:dirt": 4, "minecraft:torch": 31, "minecraft:crafting_table": 1, "minecraft:bread": 10, "minecraft:cobblestone": 58, "minecraft:bow": 1, "minecraft:arrow": 62}, # You should pay attention to the format of the inventory, and may not repeatedly give inventory items and armor/weapon items, for example, **if your set armor to minecraft:iron_pickaxe, you should not give another pickaxe in the inventory**,\n'
    prompt += '    "underground": 30, # the number of blocks to dig down, if the block ususally generates on the surface, you can set it to 0,\n'
    prompt += '    "armor": {"armor.head": "", "armor.chest": "", "armor.legs": "", "armor.feet": "", "weapon.mainhand": "minecraft:stone_pickaxe", "weapon.offhand": ""}, # the armor and weapon of the agent, you should pay attention to the format of the armor. If you want to add enchantments to the weapon, you can add it like this: "weapon.mainhand": "minecraft:stone_pickaxe{Enchantments:[{id:silk_touch, lvl:1}]}",\n'
    prompt += '    "generate_block": True, # whether to use command to generate a new block. If the block natrually generates in front of your with a great probability, say 95%, you can set it to False, for example, if you want to mine a sand/wood block and tp to a desert/birch forest biome since sand/wood block usually can be seen. However for iron ore, you should set it to True, since iron ore is rare and may not be found. Or for items like diamond blocks/crafting table which are not naturally generated, set it to True.If you are not sure about this, you can set it to True,\n'
    prompt += '}\n'
    prompt += "```\n"
    prompt += "If the object can not be founded on the given conditions, please output 'Not suitable for mining' (for example maybe the object is in another Minecraft version than 1.16.5).\n"
    prompt += "To be precise, you should try to keep essential requirements for mining the object, for example, you should give the player a stone or better pickaxe to mine the iron ore.\n"
    prompt += "To be diverse, you should try to alter the environment configs, for example, for mining iron ore, you can change the time, weather, biome, underground level (but be reasonable), or the inventory and armor of the agent.\n"
    prompt += "To be close to real game play, think about and imitate a player mining the object in the game, what should the player prepare before mining the object? For example, for mining iron ore, the player may carry some torches, a crafting table, and some dirt blocks, ... Try your best to make the inventory more reasonable and diverse.\n"
    prompt += f"Now please output the environment configs for mining the object {mine} in json format given the relationships and properties of the object.\n"
    prompt += f"The properties of the object are:\n"
    prompt += f"{response['properties']}\n"
    prompt += f"The relationships of the object are:\n"
    prompt += f"{response['relationships']}\n"
    prompt += "Now output the config or 'Not suitable for mining'. DO NOT output anything else.\n"
    prompt += "**Again, do not generate biome that does not exist, or generate inventory items that are same as armor/weapon items. And generate as much similar to the real game play as possible. If we can't see a block when we are teleported randomly to a position at high prob, the `generate_block` should be true (which should be a default option). Also, if the underground level is not specified, you can set it according to your own experience.**\n"
    
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a Minecraft AI assistant and expert."},
            {"role": "user", "content": prompt}
        ]
    })

    headers = {
        'Authorization': f'{openai_key}',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.text
    # replace null with None
    response = response.replace('null', 'None')
    response = response.replace('true', 'True')
    response = response.replace('false', 'False')
    result = eval(response)['choices'][0]['message']['content']

    if result == 'Not suitable for mining':
        continue

    # only keep things inside of {}
    result = result[result.find('{'):result.rfind('}')+1]
    result = result.replace('True', 'true')
    result = result.replace('False', 'false')

    os.makedirs('configs', exist_ok=True)

    with open(f'configs/{name}.json', 'w') as f:
        f.write(result)