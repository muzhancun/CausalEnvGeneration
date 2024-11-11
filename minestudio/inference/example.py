
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback
from minestudio.models import OpenAIPolicy, load_openai_policy

if __name__ == '__main__':
    
    policy = load_openai_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights"
    ).to("cuda")
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="forest", 
        callbacks=[
            RecordCallback(record_path="./output", fps=30),
            SpeedTestCallback(50),
        ]
    )
    memory = None
    obs, info = env.reset()
    for i in range(1200):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
