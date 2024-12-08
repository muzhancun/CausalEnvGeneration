<!--
 * @Date: 2024-11-29 15:45:12
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-11-29 16:24:27
 * @FilePath: /MineStudio/docs/source/simulator/design-principles.md
-->

# Design Principles

## Simulator Lifecycle

The simulator lifecycle is divided into three stages: `reset`, `step`, and `close`. 

- `reset`: This method is called when the environment is initialized. It returns the initial observation and information. 
    Our simulator wrapper's `reset` method code looks like this:
    ```python
    def reset(self):
        reset_flag = True
        for callback in self.callbacks:
            reset_flag = callback.before_reset(self, reset_flag)
        ... # some other code
        if reset_flag:
            obs, info = self.env.reset()
        else:
            obs, info = ...
        for callback in self.callbacks:
            obs, info = callback.after_reset(self, obs, info)
        return obs, info
    ```

    ```{hint}
    We can use callbacks to preprocess the ``obs`` or ``info`` before it is returned to the agent. 

    For example, we can add task information to the observation when the environment is reset, so that the agent knows what task it is going to perform.  

    Besides, we can implement fast reset by suppressing the internal environment reset.
    ```

- `step`: This method is called when the agent takes an action. It returns the observation, reward, termination status, and information. 
    The `step` method code looks like this:
    ```python
    def step(self, action):
        for callback in self.callbacks:
            action = callback.before_step(self, action)
        obs, reward, terminated, truncated, info = self.env.step(action.copy()) 
        ... # some other code
        for callback in self.callbacks:
            obs, reward, terminated, truncated, info = callback.after_step(
                self, obs, reward, terminated, truncated, info
            )
        return obs, reward, terminated, truncated, info
    ```
    ```{hint}
    We can use callbacks to preprocess the action before it is passed to the environment. For example, we can mask the action that we do not want to use. 
    
    Or we can use callbacks to post-process the observation, reward, termination status, and information before the environment returns them. 

    The callbacks can be sequentially executed in the order they are added to the simulator. 
    ```

- `close`: This method is called when the environment is closed.
    The `close` method code looks like this:
    ```python
    def close(self):
        for callback in self.callbacks:
            callback.before_close(self)
        close_status = self.env.close()
        for callback in self.callbacks:
            callback.after_close(self)
        return close_status
    ```

    ```{hint}
    We can use callbacks to do some cleanup work before the environment is closed. For example, we can save the trajectories or doing some logging. 
    ```

## Callbacks

Callbacks are used to customize the environment. All the callbacks are optional, and you can use them in any combination. 

The structure of a callback is as follows:
```python
class MinecraftCallback:
    
    def before_step(self, sim, action):
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        return obs, reward, terminated, truncated, info
    
    def before_reset(self, sim, reset_flag: bool) -> bool: # whether need to call env reset
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        return obs, info
    
    def before_close(self, sim):
        return
    
    def after_close(self, sim):
        return
    
    def before_render(self, sim):
        return
    
    def after_render(self, sim):
        return
```