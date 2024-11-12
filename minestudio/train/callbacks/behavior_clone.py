from minestudio.train.callbacks.callback import ObjectiveCallback

class BehaviorCloneCallback(ObjectiveCallback):
        
        def __init__(self):
            super().__init__()
        
        def __call__(self, batch, batch_idx, step_name, latents, mine_policy):
            return {}