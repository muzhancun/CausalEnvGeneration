'''
Date: 2024-11-25 07:36:30
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 08:07:42
FilePath: /MineStudio/minestudio/inference/summarizer/base_summarizer.py
'''
from abc import abstractmethod
from typing import List, Dict

class EpisodeSummarizer:

    def __init__(self):
        pass
    
    @abstractmethod
    def summarize(self, episodes: List) -> Dict:
        raise NotImplementedError