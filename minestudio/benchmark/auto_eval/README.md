
# Automatic evaluation pipeline
If you want to generate criteria files for your own tasks:
```bash
cd auto_eval/
python rule_generation.py
```

## Evaluating videos with VLM
```bash
#Compare two videos
python video_comparison.py --video_path_a='xxx/xxx.mp4' --video_path_b='xxx/xxx.mp4' --criteria_path='xxx/xxx.txt' 
#Individual video evaluation
python individual_video_rating.py --video_path='xxx/xxx/' --criteria_path='xxx/xxx/' 
#Batch video evaluation
python batch_video_rating.py --videos_path='xxx/xxx/' --criteria_files_path='xxx/xxx/' 
```

For batch video rating, you should organize video files and criteria files with the following structure:

```
videos_path     
├── build_waterfall # task_name_1     
│     ├── xxx.mp4
│     ├── xxx.mp4
├── build_house # task_name_2
│     ├── xxx.mp4
│     ├── xxx.mp4
├── xxx # task_name_3
│     ├── xxx.mp4
│     ├── xxx.mp4
```

```
criteria_files_path     
├── build_waterfall.txt # task_name_1     
├── build_house.txt # task_name_2
├── xxx.txt # task_name_3
```




