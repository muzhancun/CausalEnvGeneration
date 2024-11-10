'''
Date: 2024-11-10 10:06:28
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 14:25:53
FilePath: /MineStudio/minestudio/data/minecraft/utils.py
'''
import av
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional

def write_video(
    file_name: str, 
    frames: Sequence[np.ndarray], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def batchify(batch_in: Sequence[Dict[str, Any]]) -> Any:
    example = batch_in[0]
    if isinstance(example, Dict):
        batch_out = {
            k: batchify([item[k] for item in batch_in]) \
                for k in example.keys()
        }
    elif isinstance(example, torch.Tensor):
        batch_out = torch.stack(batch_in, dim=0)
    elif isinstance(example, int):
        batch_out = torch.tensor(batch_in, dtype=torch.int32)
    elif isinstance(example, float):
        batch_out = torch.tensor(batch_in, dtype=torch.float32)
    else:
        batch_out = batch_in
    return batch_out

class MineDistributedBatchSampler(Sampler):

    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_replicas=None, # num_replicas is the number of processes participating in the training
        rank=None,         # rank is the rank of the current process within num_replicas
        shuffle=False, 
        drop_last=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        assert shuffle is False, "shuffle must be False in sampler."
        assert drop_last is True, "drop_last must be True in sampler."
        print(f"{rank = }, {num_replicas = }")
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_total_samples = len(self.dataset)
        self.num_samples_per_replica = self.num_total_samples // num_replicas
        replica_range = (self.num_samples_per_replica * rank, self.num_samples_per_replica * (rank + 1)) # [start, end)
        
        num_past_samples = 0
        episodes_within_replica = [] # (episode, epsode_start_idx, episode_end_idx, item_bias)
        self.episodes_with_items = self.dataset.episodes_with_items
        for episode, length, item_bias in self.episodes_with_items:
            if num_past_samples + length > replica_range[0] and num_past_samples < replica_range[1]:
                episode_start_idx = max(0, replica_range[0] - num_past_samples)
                episode_end_idx = min(length, replica_range[1] - num_past_samples)
                episodes_within_replica.append((episode, episode_start_idx, episode_end_idx, item_bias))
            num_past_samples += length
        self.episodes_within_replica = episodes_within_replica

    def __iter__(self):
        """
        Build batch of episodes, each batch is consisted of `self.batch_size` episodes.
        Only if one episodes runs out of samples, the batch is filled with the next episode.
        """
        batch = []
        reading_episodes = [ None for _ in range(self.batch_size) ]
        next_episode_idx = 0
        while True:
            # feed `reading_episodes` with the next episode
            for i in range(self.batch_size):
                if reading_episodes[i] is None:
                    if next_episode_idx >= len(self.episodes_within_replica):
                        break
                    reading_episodes[i] = self.episodes_within_replica[next_episode_idx]
                    next_episode_idx += 1
            # use while loop to build batch
            while len(batch) < self.batch_size:
                record_batch_length = len(batch)
                for i in range(self.batch_size):
                    if reading_episodes[i] is not None:
                        episode, start_idx, end_idx, item_bias = reading_episodes[i]
                        batch.append(item_bias + start_idx)
                        if start_idx+1 < end_idx:
                            reading_episodes[i] = (episode, start_idx + 1, end_idx, item_bias)
                        else:
                            reading_episodes[i] = None
                if record_batch_length == len(batch):
                    # stop iteration if no new samples are added to the batch
                    return None
            yield batch
            batch = []

    def __len__(self):
        return self.num_samples_per_replica // self.batch_size