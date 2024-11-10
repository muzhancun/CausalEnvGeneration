'''
Date: 2024-11-10 10:06:28
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 11:42:43
FilePath: /MineStudio/minestudio/data/minecraft/utils.py
'''
import av
import cv2
import numpy as np
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

def flatten_collate_dict(
    dic: List[Dict[str, np.ndarray]], support_keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Flatten dict of tensors into tensor of dict."""
    result = dict()
    if support_keys is None:
        support_keys = dic[0].keys()
    for key, val in dic[0].items():
        if key not in support_keys:
            continue
        if isinstance(val, np.ndarray):
            result[key] = np.stack([d.get(key, np.zeros_like(val)) for d in dic], axis=0)
        else:
            result[key] = [d.get(key, None) for d in dic]
    return result

def list_to_dict(
    list_of_dict: List[Dict],
) -> Dict[str, List]:
    dict_of_list = dict()
    for key, val in list_of_dict[0].items():
        if isinstance(val, Dict):
            dict_of_list[key] = list_to_dict([d[key] for d in list_of_dict])
        else:
            dict_of_list[key] = [d[key] for d in list_of_dict]
    return dict_of_list

def collate_fn(result_in: Sequence[Mapping[str, np.ndarray]]) -> Mapping[str, Sequence[np.ndarray]]:
    """Convert sequence of dict into dict of sequence."""
    
    result_out = dict()
    example = result_in[0]
    if 'image' in example:
        tensors = [dic['image'] for dic in result_in]
        result_out['image'] = np.stack(tensors, axis = 0)
    
    action_names = ['env_action', 'agent_action', 'env_prev_action', 'agent_prev_action']
    for key in action_names:
        if key in example:
            result_out[key] = flatten_collate_dict(
                [dic[key] for dic in result_in], support_keys=None
            )
    
    if 'contractor_info' in example:
        result_out['contractor_info'] = flatten_collate_dict(
            [dic['contractor_info'] for dic in result_in], support_keys=None
        )

    if 'mask' in example:
        tensors = [dic['mask'] for dic in result_in]
        result_out['mask'] = np.stack(tensors, axis = 0)
    
    if 'text' in example:
        texts = [dic['text'] for dic in result_in]
        result_out['text'] = texts

    if 'obs_conf' in example:
        result_out['obs_conf'] = list_to_dict(
            [dic['obs_conf'] for dic in result_in]
        )
    
    if 'segment' in example:
        result_out['segment'] = flatten_collate_dict(
            [ dic['segment'] for dic in result_in ], support_keys=None
        )
    
    return result_out

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
        
        self.batch_size = batch_size
        self.num_total_samples = len(dataset)
        self.num_samples_per_replica = self.num_total_samples // num_replicas
        replica_range = (self.num_samples_per_replica * rank, self.num_samples_per_replica * (rank + 1)) # [start, end)
        
        num_past_samples = 0
        episodes_within_replica = [] # (episode, epsode_start_idx, episode_end_idx, item_bias)
        self.episodes_with_items = dataset.episodes_with_items
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