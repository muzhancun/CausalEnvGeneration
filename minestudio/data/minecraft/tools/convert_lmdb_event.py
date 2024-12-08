'''
Date: 2024-11-10 12:26:39
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:29:04
FilePath: /MineStudio/minestudio/data/minecraft/tools/convert_lmdb_event.py
'''

import re
import os
import lmdb
import time
import redis
import random
import pickle
import argparse
import shutil
from collections import OrderedDict

import math 
import torch
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from rich import print
from rich.console import Console
from yaml import load, CLoader
from pathlib import Path
from typing import Union, Tuple, List, Dict

from minestudio.data.minecraft.core import Kernel

'''
    Desired data structure of lmdb files: 
    {
        '__num_events__': 600, 
        '__event_info__': {
            'pickup:mutton': {
                '__num_items__': 500,
                '__num_episodes__': 20,
            }, 
            'mine_block:grass': {
                '__num_items__': 450,
                '__num_episodes__': 16,
            }, 
            ...
        }, 
        '(pickup:mutton, 0)': (episode_xxx, t1, v1), 
        '(pickup:mutton, 1)': (episode_xxx, t2, v2), 
        '(pickup:mutton, 2)': (episode_yyy, t1, v1),
        '(mine_block:grass, 0)': (episode_zzz, t1, v1),
        ...
    }
'''

def main(args):
    
    event_path = Path(args.input_dir) / "event"
    if event_path.is_dir():
        print(f"Directory {event_path} exists, remove and recreate one. ")
        shutil.rmtree(event_path)
    event_path.mkdir(parents=True)
    
    contractor_info_path = Path(args.input_dir) / 'contractor_info'
    assert contractor_info_path.is_dir(), f"Directory {contractor_info_path} does not exist. "
    
    kernel = Kernel(
        dataset_dirs=[args.input_dir], 
        enable_video=True, 
        enable_action=True,
        enable_contractor_info=True,
    )
    
    episode_with_length = kernel.get_episodes_with_length()
    episodes = [x for x in episode_with_length.keys()]
    
    events = {}
    # monitor_fields = ['delta_craft_item', 'delta_mine_block', 'delta_pickup']
    monitor_fields = ['events']
    for episode in tqdm(episodes):
        length = episode_with_length[episode]
        frames, mask = kernel.read_frames(episode, start=0, win_len=length, skip_frame=1, source_type='contractor_info')
        assert mask.sum() == length, f"Mask sum: {mask.sum()}, length: {length}. "
        # enumerate all fields of interest and generate all the events
        for field in monitor_fields:
            records: List[Dict] = frames[field]
            for t, record in enumerate(records):
                if len(record) == 0:
                    continue
                for event, value in record.items():
                    if event not in events:
                        events[event] = {}
                    if episode not in events[event]:
                        events[event][episode] = []
                    events[event][episode].append( (t, value) )
    
    # write events into lmdb files in the desired structure
    lmdb_data = {
        '__num__events__': len(events), 
        '__event_info__': {}, 
    }
    
    print(lmdb_data['__num__events__'])
    
    for event, episode_items in events.items():
        lmdb_data['__event_info__'][event] = {
            '__num_episodes__': len(episode_items),
            '__num_items__': sum([len(x) for x in episode_items.values()]),
        }
    
    for event, episode_items in events.items():
        event_item_id = 0
        for episode, items in episode_items.items():
            for e_time, value in items:
                key = str((event, event_item_id))
                lmdb_data[key] = (episode, e_time, value)
                event_item_id += 1
    
    with lmdb.open(str(event_path), map_size=1<<40) as env:
        with env.begin(write=True) as txn:
            for key, value in lmdb_data.items():
                key = key.encode()
                value = pickle.dumps(value)
                txn.put(key, value)
    
    print(f"Write lmdb data into {event_path}. ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing lmdb-format accomplishments. ")
    args = parser.parse_args()
    
    main(args)