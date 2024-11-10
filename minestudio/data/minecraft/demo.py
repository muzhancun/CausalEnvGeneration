'''
Date: 2024-11-10 11:01:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:02:26
FilePath: /MineStudio/minestudio/data/minecraft/demo.py
'''
import os
import av
import cv2
import time
import hydra
import random
import string
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional

from minestudio.data.minecraft.part_event import EventDataset
from minestudio.data.minecraft.part_raw import RawDataset
from minestudio.data.minecraft.dataset import MinecraftDataset
from minestudio.data.minecraft.utils import MineDistributedBatchSampler, write_video, collate_fn

def write_to_frame(frame: np.ndarray, txt: str, row: int, col: int, color=(255, 0, 0)) -> None:
    cv2.putText(frame, txt, (col, row), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 1)

def dump_trajectories(
    dataloader, 
    num_samples: int = 1, 
    save_fps: int = 20, 
    **kwargs
) -> None:
    
    def un_batchify_actions(actions_in: Dict[str, torch.Tensor]) -> List[Dict]:
        actions_out = []
        for bidx in range(len(actions_in['attack'])):
            action = {}
            for k, v in actions_in.items():
                action[k] = v[bidx].numpy()
            actions_out.append(action)
        return actions_out
    
    traj_dir = Path("./traj_dir")
    video_dir = traj_dir / "videos"
    action_dir = traj_dir / "actions"
    video_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)
    for idx, data in enumerate(tqdm(dataloader)):
        if idx > num_samples: break
        image = data['img']
        action = data['action']
        action = un_batchify_actions(action)
        B, T = image.shape[:2]
        for i in range(B):
            vid = ''.join(random.choices(string.ascii_letters + string.digits, k=11))
            write_video(
                file_name=str(video_dir / f"{vid}.mp4"),
                frames=image[i].numpy().astype(np.uint8),
            )
            with open(action_dir / f"{vid}.pkl", 'wb') as f:
                pickle.dump(action[i], f)

def read_dataloader(
    dataloader, 
    num_samples: int = 1, 
    resolution: Tuple[int, int] = (320, 180), 
    legend: bool = False,
    temporal_mask: bool = False,
    save_fps: int = 20, 
    **kwargs,
) -> None:
    frames = []
    for idx, data in enumerate(tqdm(dataloader)):
        # continue
        if idx > num_samples:
            break
        action = data['env_action']
        prev_action = data.get("env_prev_action", None)
        image = data['image']
        text = data['text']

        color = (255, 0, 0)
        for bidx, (tframes, txt) in enumerate(zip(image, text)):
            cache_frames = []
            for tidx, frame in enumerate(tframes):
                if 'segment' in data:
                    COLORS = [
                        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
                        (255, 255, 0), (255, 0, 255), (0, 255, 255),
                        (255, 255, 255), (0, 0, 0), (128, 128, 128),
                        (128, 0, 0), (128, 128, 0), (0, 128, 0),
                        (128, 0, 128), (0, 128, 128), (0, 0, 128),
                    ]
                    obj_id = data['segment']['obj_id'][bidx][tidx].item()
                    if obj_id != -1:
                        segment_mask = data['segment']['obj_mask'][bidx][tidx]
                        if isinstance(segment_mask, torch.Tensor):
                            segment_mask = segment_mask.numpy()
                        colors = np.array(COLORS[obj_id]).reshape(1, 1, 3)
                        segment_mask = (segment_mask[..., None] * colors).astype(np.uint8)
                        segment_mask = segment_mask[:, :, ::-1] # bgr -> rgb
                        frame = cv2.addWeighted(frame, 1.0, segment_mask, 0.5, 0.0)
                
                if legend:
                    cv2.putText(frame, f"time: {tidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    cv2.putText(frame, txt, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    
                    if 'contractor_info' in data:
                        try:
                            pitch = data['contractor_info']['pitch'][bidx][tidx]
                            yaw = data['contractor_info']['yaw'][bidx][tidx]
                            cursor_x = data['contractor_info']['cursor_x'][bidx][tidx]
                            cursor_y = data['contractor_info']['cursor_y'][bidx][tidx]
                            isGuiInventory = data['contractor_info']['isGuiInventory'][bidx][tidx]
                            isGuiOpen = data['contractor_info']['isGuiOpen'][bidx][tidx]
                            cv2.putText(frame, f"Pitch: {pitch:.2f}", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"Yaw: {yaw:.2f}", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"isGuiOpen: {isGuiOpen}", (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"isGuiInventory: {isGuiInventory}", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"CursorX: {cursor_x:.2f}", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"CursorY: {cursor_y:.2f}", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        except:
                            cv2.putText(frame, f"No Contractor Info", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    act = {k: v[bidx][tidx].numpy() for k, v in action.items()}
                    if prev_action is not None:
                        pre_act = {k: v[bidx][tidx].numpy() for k, v in prev_action.items()}
                    for row, ((k, v), (_, pv)) in enumerate(zip(act.items(), pre_act.items())):
                        if k != 'camera':
                            v = int(v.item())
                            pv = int(pv.item())
                        cv2.putText(frame, f"{k}: {v}({pv})", (10, 45 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cache_frames.append(frame.astype(np.uint8))
            
            frames = frames + cache_frames
    
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    file_name = f"save_{timestamp}.mp4"
    write_video(file_name, frames, fps=save_fps, width=resolution[0], height=resolution[1])

def visualize_raw_dataset(args):
    raw_dataset = RawDataset(
        dataset_dirs=args.dataset_dirs, 
        enable_video=args.enable_video, 
        enable_action=args.enable_action, 
        enable_contractor_info=args.enable_contractor_info,
        enable_segment=args.enable_segment,
        win_len=args.win_len, 
        skip_frame=args.skip_frame,
        frame_width=args.frame_width, 
        frame_height=args.frame_height,
        enable_augmentation=args.enable_augmentation,
        enable_resize=args.enable_resize,
    )
    Console().log(f"num-workers: {args.num_workers}")
    batch_sampler = MineDistributedBatchSampler(
        dataset=raw_dataset,
        batch_size=args.batch_size,
        num_replicas=1, 
        rank=0,
        shuffle=args.shuffle,
    )
    dataloader = DataLoader(
        dataset=raw_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    read_dataloader(
        dataloader, 
        num_samples=args.num_samples, 
        resolution=(args.frame_width, args.frame_height), 
        legend=args.legend,
        save_fps=args.save_fps,
    )

def visualize_event_dataset(args):
    event_dataset = EventDataset(
        dataset_dirs=args.dataset_dirs, 
        enable_video=args.enable_video, 
        enable_action=args.enable_action, 
        enable_contractor_info=args.enable_contractor_info,
        enable_segment=args.enable_segment,
        win_len=args.win_len, 
        skip_frame=args.skip_frame, 
        frame_width=args.frame_width, 
        frame_height=args.frame_height, 
        enable_resize=args.enable_resize,
        enable_augmentation=args.enable_augmentation,
        event_regex=args.event_regex, 
        min_nearby=args.min_nearby,
        max_within=args.max_within,
        bias=args.bias, 
    )
    Console().log(f"num-workers: {args.num_workers}")
    dataloader = DataLoader(
        dataset=event_dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=collate_fn,
    )
    
    # dump_trajectories(
    read_dataloader(
        dataloader, 
        num_samples=args.num_samples, 
        resolution=(args.frame_width, args.frame_height), 
        legend=args.legend,
        save_fps=args.save_fps,
    )

@hydra.main(config_path="demo_configs", config_name="type-raw")
def main(args):
    if args.dataset_type == 'event':
        visualize_event_dataset(args)
    elif args.dataset_type == 'raw':
        visualize_raw_dataset(args)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

if __name__ == '__main__':
    main()