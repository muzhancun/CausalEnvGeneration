<!--
 * @Date: 2024-11-28 22:13:52
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-11-29 13:56:21
 * @FilePath: /MineStudio/docs/source/overview/installation.md
-->
(gentle-intro)=
# Installation

```{note}
If you encounter any issues during installation, please open an issue on [GitHub](https://github.com/phython96/MineStudio/issues). 
```

Welcome to MineStudio, please follow the tutorial below for installation.

## Install JDK 8
To ensure that the Simulator runs smoothly, please make sure that JDK 8 is installed on your system. We recommend using conda to maintain an environment on Linux systems. 
```console
$ conda create -n minestudio python=3.10 -y
$ conda activate minestudio
$ conda install --channel=conda-forge openjdk=8 -y
```

## Install MineStudio

a. Install MineStudio from the [GitHub](https://github.com/phython96/MineStudio). 
```console
$ pip install git+https://github.com/phython96/MineStudio.git
```

b. Install MineStudio from [PyPI](https://pypi.org/project/minestudio/). 
```console
$ pip install minestudio
```

## Install the rendering tool
For users with *nvidia graphics cards*, we recommend installing **VirtualGL**; for other users, we recommend using **Xvfb**, which supports CPU rendering but is relatively slower. 

```{note}
Installing rendering tools may require **root** permissions. 
```
There are two options: 
``````{tab-set}

`````{tab-item} Xvfb
```console
$ apt update 
$ apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev 
```
`````

`````{tab-item} VirtualGL
```{warning}
Not all graphics cards support virtualGL. If you do not have speed requirements, it is recommended to use the easier-to-install xvfb rendering tool. 
```

You need to download the following sources: 
- [virtualgl_3.1_amd64.deb](https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download)
- [vgl_entrypoint.sh]()

```console
$ apt update 
$ apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev 
```
Install the downloaded package.
```console
$ dpkg -i virtualgl_3.1_amd64.deb
```
Shutdown the display manager and configure VirtualGL.
```console
$ service gdm stop 
```
Configure VirtualGL.
```console
$ /opt/VirtualGL/bin/vglserver_config 
```
```{note}
First choose 1，then Yes, No, No, No，finally enter X
```
Start the display manager.
```console
$ service gdm start
```
Start the VirtualGL server.
```console
$ bash vgl_entrypoint.sh
```
```{warning}
Each time the system is restarted, it may be necessary to run ``vgl_entrypoint.sh``.
```

Configure the environment variables. 
```console
$ export PATH="${PATH}:/opt/VirtualGL/bin" 
$ export LD_LIBRARY_PATH="/usr/lib/libreoffice/program:${LD_LIBRARY_PATH}" 
$ export VGL_DISPLAY="egl" 
$ export VGL_REFRESHRATE="$REFRESH"
$ export DISPLAY=:1
```

`````

``````

## Verify by running simulator

```{hint}
The first time you run it, the script will ask whether to download the compiled model from Hugging Face; just choose Y. 
```

If you are using **Xvfb**, run the following command: 
```console
$ python -m minestudio.simulator.entry
```
If you are using **VirtualGL**, run the following command: 
```console
$ MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry
```

If you see the following output, the installation is successful. 
```
Speed Test Status: 
Average Time: 0.03 
Average FPS: 38.46 
Total Steps: 50 

Speed Test Status: 
Average Time: 0.02 
Average FPS: 45.08 
Total Steps: 100 
```