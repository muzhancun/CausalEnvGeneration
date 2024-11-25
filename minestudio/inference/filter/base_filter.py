'''
Date: 2024-11-25 07:36:18
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 07:38:19
FilePath: /MineStudio/minestudio/inference/filter/base_filter.py
'''
from abc import abstractmethod
from typing import List, Dict

class EpisodeFilter:

    def __init__(self):
        pass

    @abstractmethod
    def filter(self, episodes: List) -> List:
        raise NotImplementedError