'''
Date: 2024-11-25 08:11:33
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 08:18:44
FilePath: /MineStudio/minestudio/tutorials/inference/example.py
'''
import ray
import time
from minestudio.inference import EpisodePipeline
from minestudio.inference.generator import EpisodeGenerator
from minestudio.inference.filter import EpisodeFilter
from minestudio.inference.summarizer import EpisodeSummarizer
from minestudio.inference.recorder import EpisodeRecorder

class MyGenerator(EpisodeGenerator):
    def generate(self):
        for i in range(10):
            time.sleep(0.5)
            yield i

class MyFilter(EpisodeFilter):
    def filter(self, episodes):
        return [e for e in episodes if e % 2 == 0]

class MySummarizer(EpisodeSummarizer):
    def summarize(self, episodes):
        return episodes, {"sum": sum(episodes)}

class MyRecoder(EpisodeRecorder):
    def record(self, episodes, summary):
        print(episodes, summary)

if __name__ == '__main__':
    pipeline = EpisodePipeline(
        episode_generator=MyGenerator(),
        episode_filter=MyFilter(),
        episode_summarizer=MySummarizer(),
        episode_recoder=MyRecoder(),
    )
    pipeline.run()