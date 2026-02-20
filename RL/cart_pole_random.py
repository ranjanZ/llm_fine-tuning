import gymnasium as gym
import time

def action(env):
    return env.action_space.sample()

def render_episode(env, episode_ended=False):
    """Handle all rendering - both during episode and after it ends"""
    env.render()
    time.sleep(0.02)
    
    if episode_ended:
        print("   Watching fall...")
        for _ in range(20):
            state, reward, done, truncated, _ =env.step(action(env))
            #print("reward",reward,done)
            env.render()
            time.sleep(0.05)




# Main loop
for ep in range(1, 21):
    # Create environment with or without rendering
    if ep % 4 == 0:
        env = gym.make('CartPole-v1', render_mode='human')
        print(f"\n▶️ Episode {ep}")
    else:
        env = gym.make('CartPole-v1')
    
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        # Take a step
        state, reward, done, truncated, _ = env.step(action(env))
        total_reward += reward
        
        #print("reward",reward,done)
        # Single render call - passes whether episode ended
        if ep % 4 == 0:
            render_episode(env, done or truncated)
        
        if done or truncated:
            print(f"total_reward:{total_reward}")
            break
    
    env.close()

print("\n✅ Done!")


