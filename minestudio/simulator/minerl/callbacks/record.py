'''
Date: 2024-11-11 16:40:57
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-11 17:02:30
FilePath: /MineStudio/scratch/caishaofei/workspace/MineStudio/minestudio/simulator/minerl/callbacks/record.py
'''
import av
import cv2
from pathlib import Path
from minestudio.simulator.minerl.callbacks.callback import MinecraftCallback

class RecordCallback(MinecraftCallback):
    
    def __init__(self, record_path: str, fps: int = 20):
        super().__init__()
        self.record_path = Path(record_path)
        self.record_path.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.episode_id = 0
        self.frames = []
    
    def before_reset(self, sim, reset_flag: bool) -> bool:
        self._save_episode()
        self.episode_id += 1
        return reset_flag
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.frames.append(info['pov'])
        return obs, reward, terminated, truncated, info
    
    def before_close(self, sim):
        self._save_episode()
    
    def _save_episode(self):
        if len(self.frames) == 0:
            return 
        output_path = self.record_path / f'episode_{self.episode_id}.mp4'
        with av.open(output_path, mode="w", format='mp4') as container:
            stream = container.add_stream("h264", rate=self.fps)
            stream.width = self.frames[0].shape[1]
            stream.height = self.frames[0].shape[0]
            for frame in self.frames:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        print(f'Episode {self.episode_id} saved at {output_path}')
        self.frames = []