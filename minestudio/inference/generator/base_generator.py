'''
Date: 2024-11-25 07:36:18
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 07:38:19
FilePath: /MineStudio/minestudio/inference/generator/base_generator.py
'''
from abc import abstractmethod
from typing import List, Dict

class EpisodeGenerator:
    
    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass