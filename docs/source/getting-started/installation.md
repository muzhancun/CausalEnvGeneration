<!--
 * @Date: 2024-11-28 22:13:52
 * @LastEditors: muzhancun muzhancun@126.com
 * @LastEditTime: 2024-11-28 22:14:19
 * @FilePath: /MineStudio/docs/source/getting-started/installation.md
-->
# Installing Minestudio

```bash
conda create -n minestudio python=3.10 -y
conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
git clone git@github.com:phython96/MineStudio.git
cd MineStudio
pip install -e .

cd minestudio/simulator/minerl
FORCE_CPU_RENDER=1 python entry.py
```