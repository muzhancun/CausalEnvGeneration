<!--
 * @Date: 2024-11-29 15:45:12
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-11-29 16:10:33
 * @FilePath: /MineStudio/docs/source/simulator/design-principles.md
-->

# Design Principles

## Simulator Lifecycle

The simulator lifecycle is divided into three stages: `reset`, `step`, and `close`. 

- `reset`: This method is called when the environment is initialized. It returns the initial observation and information. 
    Our simulator wrapper's `reset` method code looks like this:
    ```python
    def reset(self):
        for callback in self.callbacks:
            callback.before_reset(self)
        ... # some other code
        obs, info = self.env.reset()
        for callback in self.callbacks:
            obs, info = callback.after_reset(self, obs, info)
        return obs, info
    ```

    ```{hint}
    We can use callbacks to preprocess the ``obs`` or ``info`` before it is returned to the agent. 

    For example, we can add task information to the observation when the environment is reset, so that the agent knows what task it is going to perform.  
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