# MineStudio
A simple and efficient Minecraft development kit for AI research.

```sh
conda create -n minestudio python=3.10 -y
conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
git clone git@github.com:phython96/MineStudio.git
cd MineStudio
pip install -e .

cd minestudio/simulator/minerl
FORCE_CPU_RENDER=1 python entry.py
```