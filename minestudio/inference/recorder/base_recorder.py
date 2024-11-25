'''
Date: 2024-11-25 07:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 08:19:06
FilePath: /MineStudio/minestudio/inference/recorder/base_recorder.py
'''
from abc import abstractmethod
from typing import List, Dict

class EpisodeRecorder:

    def __init__(self):
        pass

    @abstractmethod
    def record(self, episodes: List, summary: Dict):
        raise NotImplementedError