'''
Date: 2024-11-25 07:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 08:34:51
FilePath: /MineStudio/minestudio/inference/recorder/base_recorder.py
'''
from abc import abstractmethod
from typing import List, Dict

class EpisodeRecorder:

    def __init__(self):
        pass

    def record(self, episodes: List, summary: Dict):
        print(f"Summary of {len(episodes)} episodes: {summary}")
        return "Vanilla Recorder"