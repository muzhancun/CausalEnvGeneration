'''
Date: 2024-11-25 07:36:30
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:10:04
FilePath: /MineStudio/minestudio/inference/summarizer/base_summarizer.py
'''
from abc import abstractmethod
from typing import List, Dict, Tuple

class EpisodeSummarizer:

    def __init__(self):
        pass

    def summarize(self, episode_generator: List) -> Tuple[List, Dict]:
        return episodes, {}