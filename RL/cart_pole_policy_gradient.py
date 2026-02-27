import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2), nn.Softmax(dim=-1))
    def forward(self, x): 
        return self.net(x)

def get_action(policy, state):
    with torch.no_grad():
        probs = policy(torch.FloatTensor(state).unsqueeze(0))[0]
        return torch.argmax(probs).item()

# Initialize
env = gym.make('CartPole-v1')
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training loop
for episode in range(200):
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = policy(state_t)[0]
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        
        state, reward, done, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = done or truncated
    
    # Calculate returns and update
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = torch.stack([-lp * r for lp, r in zip(log_probs, returns)]).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Visualize every 20 episodes
    if (episode + 1) % 10 == 0:
        print(f"\n🎬 Episode {episode+1} | Reward: {sum(rewards)}")
        # Show trained policy in action
        vis_env = gym.make('CartPole-v1', render_mode='human')
        s, _ = vis_env.reset()
        total = 0
        while True:
            a = get_action(policy, s)
            s, r, d, tr, _ = vis_env.step(a)
            total += r
            vis_env.render()
            time.sleep(0.02)
            if d or tr:
                print(f"   Test run: {total} steps")
                for _ in range(20):
                    a = get_action(policy, s)
                    s, r, d, tr, _ =vis_env.step(a)
                    #print("reward",reward,done)
                    env.render()
                    time.sleep(0.05)

                break
        vis_env.close()
        time.sleep(1)

print("\n✅ Done!")
