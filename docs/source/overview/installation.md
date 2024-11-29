<!--
 * @Date: 2024-11-28 22:13:52
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-11-29 10:35:53
 * @FilePath: /MineStudio/docs/source/overview/installation.md
-->
(gentle-intro)=
# Installation

```{note}
If you encounter any issues during installation, please open an issue on [GitHub](https://github.com/phython96/MineStudio/issues). 
```

Welcome to MineStudio, please follow the tutorial below for installation.

1. To ensure that the Simulator runs smoothly, please make sure that JDK 8 is installed on your system. We recommend using conda to maintain an environment on Linux systems. 
    ```console
    $ conda create -n minestudio python=3.10 -y
    $ conda activate minestudio
    $ conda install --channel=conda-forge openjdk=8 -y
    ```

2. Install MineStudio. 
    
    a. Install MineStudio from the github. 
    ```console
    pip install git+https://github.com/phython96/MineStudio.git
    ```

    b. Install MineStudio from [PyPI](https://pypi.org/project/minestudio/). 
    ```console
    pip install minestudio
    ```

3. Install the rendering tool. For users with **nvidia graphics cards**, we recommend installing virtualGL; for other users, we recommend using xvfb, which supports CPU rendering but is relatively slower. 

    ```{note}
    Installing rendering tools may require **root** permissions. 
    ```

    a. **Render using Xvfb.**
    ```console
    $ apt update 
    $ apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev 
    ```

    b. **Render using VirtualGL. (Optional)** 

    ```{warning}
    Not all graphics cards support virtualGL. If you do not have speed requirements, it is recommended to use the easier-to-install xvfb rendering tool. 
    ```

    You need to download the following sources: 
    - [virtualgl_3.1_amd64.deb](https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download)
    - [vgl_entrypoint.sh]()

    ```console
    $ apt update 
    $ apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev 
    $ dpkg -i virtualgl_3.1_amd64.deb
    $ service gdm stop 
    $ /opt/VirtualGL/bin/vglserver_config # first choose 1，then Yes, No, No, No，finally enter X
    $ service gdm start
    $ bash vgl_entrypoint.sh
    ```
    Configure the environment variables. 
    ```console
    $ export PATH="${PATH}:/opt/VirtualGL/bin" 
    $ export LD_LIBRARY_PATH="/usr/lib/libreoffice/program:${LD_LIBRARY_PATH}" 
    $ export VGL_DISPLAY="egl" 
    $ export VGL_REFRESHRATE="$REFRESH"
    $ export DISPLAY=:1
    ```

4. Verify by running simulator. 


