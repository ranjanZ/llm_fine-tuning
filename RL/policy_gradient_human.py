import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Deeper network for complex Humanoid control
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Mean output
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize mean head with small weights (helps initial exploration)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        
        # Learnable log standard deviation (separate for each action dimension)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)  # Start with std=0.37
        
    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
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

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def normalize_returns(returns):
    """Normalize returns for stability"""
    returns = torch.FloatTensor(returns)
    if len(returns) > 1 and returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train_humanoid():
    # Initialize environment
    env = gym.make('Humanoid-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("=" * 60)
    print("🚀 Starting Humanoid Training with Pure REINFORCE")
    print("=" * 60)
    print(f"State dimension: {state_dim} (joint angles, velocities, etc.)")
    print(f"Action dimension: {action_dim} (torques for 17 joints)")
    print(f"Action space low: {env.action_space.low[0]:.2f}")
    print(f"Action space high: {env.action_space.high[0]:.2f}")
    print("=" * 60)
    
    # Initialize policy
    policy = Policy(state_dim, action_dim, hidden_dim=512)  # Larger network for Humanoid
    optimizer = optim.Adam(policy.parameters(), lr=3e-5)  # Lower learning rate for stability
    
    # Training hyperparameters
    num_episodes = 3000
    gamma = 0.99
    max_steps = 1000
    
    # Tracking variables
    best_reward = -np.inf
    reward_history = []
    running_reward = None
    episode_times = []
    
    for episode in range(num_episodes):
        start_time = time.time()
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
        
        # Calculate returns
        returns = compute_returns(rewards, gamma)
        returns = normalize_returns(returns)
        
        # Calculate REINFORCE loss
        log_probs_tensor = torch.stack(log_probs)
        loss = -torch.sum(log_probs_tensor * returns)
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (crucial for Humanoid)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Calculate episode statistics
        total_reward = sum(rewards)
        episode_time = time.time() - start_time
        episode_times.append(episode_time)
        
        # Update running reward
        if running_reward is None:
            running_reward = total_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * total_reward
        
        reward_history.append(total_reward)
        
        # Track best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), 'best_humanoid_policy.pth')
            print(f"\n✨ New best reward: {best_reward:.2f} at episode {episode+1}")
        
        # Detailed logging every episode (Humanoid needs close monitoring)
        print(f"\n📊 Episode {episode+1:4d} | Reward: {total_reward:7.2f} | "
              f"Running: {running_reward:7.2f} | Best: {best_reward:7.2f} | "
              f"Length: {len(rewards):3d} | Loss: {loss.item():8.4f} | "
              f"Time: {episode_time:.1f}s")
        
        # Adaptive learning rate based on performance
        if (episode + 1) % 100 == 0:
            # Adjust learning rate based on recent performance
            recent_rewards = reward_history[-100:]
            avg_recent = np.mean(recent_rewards)
            
            if avg_recent < 100:  # If struggling, increase exploration
                with torch.no_grad():
                    policy.log_std.data += 0.1
                print(f"   📈 Increasing exploration: log_std = {policy.log_std.mean().item():.2f}")
            
            print(f"\n📈 100-episode average: {avg_recent:.2f}")
        
        # Visualize every 200 episodes
        if (episode + 1) % 200 == 0:
            visualize_humanoid(policy, num_episodes=1)
            print(f"   Average episode time: {np.mean(episode_times[-100:]):.2f}s")
    
    env.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Final running reward: {running_reward:.2f}")
    print(f"   Average episode time: {np.mean(episode_times):.2f}s")
    print("=" * 60)
    
    # Final visualization
    print("\n🎬 Final visualization of best policy...")
    visualize_humanoid(policy, num_episodes=3)

def visualize_humanoid(policy, num_episodes=2):
    """Visualize the trained policy in action"""
    print("\n" + "-" * 40)
    print("🎬 Visualizing trained Humanoid policy...")
    print("-" * 40)
    
    env = gym.make('Humanoid-v4', render_mode='human')
    
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
        time.sleep(2)  # Pause between episodes
    
    env.close()

def train_humanoid_with_curriculum():
    """Alternative: Train with curriculum learning (easier for Humanoid)"""
    print("\n🎯 Starting Humanoid training with curriculum...")
    
    # Start with easier task (simpler environment or lower difficulty)
    env = gym.make('Humanoid-v4', 
                   forward_reward_weight=1.0,  # Encourage forward movement
                   ctrl_cost_weight=0.1,        # Lower control cost initially
                   contact_cost_weight=0.1)      # Lower contact cost initially
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    # Training with gradually increasing difficulty
    difficulties = [0.5, 0.75, 1.0]  # Progressively harder
    
    for difficulty in difficulties:
        print(f"\n📈 Training at difficulty: {difficulty}")
        
        # Adjust environment parameters based on difficulty
        env = gym.make('Humanoid-v4',
                      forward_reward_weight=difficulty * 1.0,
                      ctrl_cost_weight=0.1 / difficulty,
                      contact_cost_weight=0.1 / difficulty)
        
        # Train for some episodes at this difficulty
        for episode in range(500):
            state, _ = env.reset()
            log_probs = []
            rewards = []
            
            for step in range(1000):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mean, log_std = policy(state_tensor)
                std = torch.exp(log_std)
                
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                log_probs.append(log_prob)
                
                action_np = action[0].cpu().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                
                state, reward, terminated, truncated, _ = env.step(action_np)
                rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            returns = compute_returns(rewards, gamma=0.99)
            returns = normalize_returns(returns)
            
            log_probs_tensor = torch.stack(log_probs)
            loss = -torch.sum(log_probs_tensor * returns)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            if (episode + 1) % 50 == 0:
                total_reward = sum(rewards)
                print(f"   Difficulty {difficulty}, Episode {episode+1}: Reward = {total_reward:.2f}")
    
    print("\n✅ Curriculum training complete!")
    return policy

if __name__ == "__main__":
    # Choose which version to run
    
    # Standard training
    train_humanoid()
    
    # Or use curriculum learning for better results
    # policy = train_humanoid_with_curriculum()
    # visualize_humanoid(policy, num_episodes=3)
