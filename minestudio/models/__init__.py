'''
Date: 2024-11-11 15:59:37
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 11:47:17
FilePath: /MineStudio/minestudio/models/__init__.py
'''
from minestudio.models.base_policy import MinePolicy
from minestudio.models.rocket_one import RocketPolicy, load_rocket_policy
from minestudio.models.openai_vpt import OpenAIPolicy, load_openai_policy