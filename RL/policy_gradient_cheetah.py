import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # For continuous actions, we output mean and log_std
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # Learnable log standard deviation (same as original code's style)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        return mean, self.log_std

def get_action(policy, state, deterministic=False):
    """Get action from policy - for visualization"""
    with torch.no_grad():
        mean, log_std = policy(torch.FloatTensor(state).unsqueeze(0))
        if deterministic:
            return mean[0].cpu().numpy()
        
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action[0].cpu().numpy()

def train_cheetah():
    # Initialize environment
    env = gym.make('HalfCheetah-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    
    # Initialize policy
    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Training loop
    num_episodes = 1000
    gamma = 0.99
    max_steps = 1000
    
    best_reward = -np.inf
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # Collect trajectory
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action distribution
            mean, log_std = policy(state_tensor)
            std = torch.exp(log_std)
            
            # Create distribution and sample action
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            
            # Store log probability
            log_prob = dist.log_prob(action).sum(dim=-1)
            log_probs.append(log_prob)
            
            # Take action in environment
            action_np = action[0].cpu().numpy()
            # Clip action to environment bounds
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            
            state, reward, terminated, truncated, _ = env.step(action_np)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Convert to tensor and normalize (for stability)
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient loss (REINFORCE)
        log_probs_tensor = torch.stack(log_probs)
        loss = -torch.sum(log_probs_tensor * returns)
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Track best model
        total_reward = sum(rewards)
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), 'best_cheetah_policy.pth')
        
        # Logging
        if (episode + 1) % 10 == 0:
            print(f"\n📊 Episode {episode+1}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Best Reward: {best_reward:.2f}")
            print(f"   Episode Length: {len(rewards)}")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Log Std: {policy.log_std.mean().item():.2f}")
        
        # Visualize every 50 episodes
        if (episode + 1) % 50 == 0:
            visualize_cheetah(policy)
    
    env.close()
    print(f"\n✅ Training complete! Best reward: {best_reward:.2f}")

def visualize_cheetah(policy, num_episodes=2):
    """Visualize the trained policy in action"""
    print("\n🎬 Visualizing trained policy...")
    env = gym.make('HalfCheetah-v4', render_mode='human')
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:  # Max steps
            # Use deterministic action for visualization
            action = get_action(policy, state, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.02)  # Slow down for viewing
            
            if terminated or truncated:
                break
        
        print(f"   Test run {ep+1}: {total_reward:.2f} reward over {steps} steps")
        time.sleep(1)
    
    env.close()


if __name__ == "__main__":
    print("🚀 Starting Cheetah Training with Pure REINFORCE")
    print("=" * 50)
    
    # Train on HalfCheetah
    train_cheetah()
    
    # Uncomment to train on Ant
    # train_ant()
