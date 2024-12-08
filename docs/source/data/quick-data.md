<!--
 * @Date: 2024-12-01 08:30:33
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-01 08:41:00
 * @FilePath: /MineStudio/docs/source/data/quick-data.md
-->
Here is a minimal example to show how we load a trajectory from the dataset. 

```python
from minestudio.data import RawDataset

dataset = RawDataset(
    dataset_dirs=['/nfs/data/contractors/dataset_7xx'], 
    enable_video=True,
    enable_action=True,
    frame_width=224, 
    frame_height=224,
    win_len=128, 
    split='train', 
    split_ratio=0.9, 
    verbose=True
)
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[08:14:15] [Kernel] Driver video load 4617 episodes.  
[08:14:15] [Kernel] Driver action load 4681 episodes. 
[08:14:15] [Kernel] episodes: 4568, frames: 65291168. 
dict_keys(['text', 'timestamp', 'episode', 'progress', 'env_action', 'agent_action', 'env_prev_action', 'agent_prev_action', 'image', 'mask'])
```

```{button-ref}  ./dataset-raw
:color: primary
:outline:
:expand:

Learn more about Raw Dataset
```

Alternatively, you can also load trajectories that have specific events, for example, loading all trajectories that contain the ``kill entity`` event. 

```python
from minestudio.data import EventDataset

dataset = EventDataset(
    dataset_dirs=['/nfs/data/contractors/dataset_7xx'], 
    enable_video=True,
    enable_action=True,
    frame_width=224, 
    frame_height=224,
    win_len=128, 
    split='train', 
    split_ratio=0.9, 
    verbose=True,
    event_regex='minecraft.kill_entity:.*'
)
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[08:19:14] [Kernel] Driver video load 4617 episodes.
[08:19:14] [Kernel] Driver action load 4681 episodes. 
[08:19:14] [Kernel] episodes: 4568, frames: 65291168. 
[08:19:14] [Event Kernel] Number of loaded events: 58. 
[08:19:14] [Event Dataset] Regex: minecraft.kill_entity:.*, Number of events: 58, number of items: 19652
dict_keys(['text', 'env_action', 'agent_action', 'env_prev_action', 'agent_prev_action', 'image', 'mask'])
```

```{button-ref}  ./dataset-event
:color: primary
:outline:
:expand:

Learn more about Event Dataset
```