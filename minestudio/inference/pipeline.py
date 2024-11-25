'''
Date: 2024-11-25 07:29:21
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 08:21:09
FilePath: /MineStudio/minestudio/inference/pipeline.py
'''

import ray

from minestudio.inference.generator.base_generator import EpisodeGenerator
from minestudio.inference.filter.base_filter import EpisodeFilter
from minestudio.inference.summarizer.base_summarizer import EpisodeSummarizer
from minestudio.inference.recorder.base_recorder import EpisodeRecorder

class EpisodePipeline:
    """
    EpisodeGenerator -> EpisodeFilter -> EpisodeSummarizer -> EpisodeRecoder
    """

    def __init__(
        self, 
        episode_generator: EpisodeGenerator,
        episode_filter: EpisodeFilter,
        episode_summarizer: EpisodeSummarizer,
        episode_recoder: EpisodeRecorder,
    ):
        self.episode_generator = episode_generator
        self.episode_filter = episode_filter
        self.episode_summarizer = episode_summarizer
        self.episode_recoder = episode_recoder

    def run(self):
        episodes = self.episode_generator.generate()
        episodes = self.episode_filter.filter(episodes)
        episodes, summary = self.episode_summarizer.summarize(episodes)
        self.episode_recoder.record(episodes, summary)
