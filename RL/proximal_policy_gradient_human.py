import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.distributions import Normal
from collections import deque
import os
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        
    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        return mean, self.log_std
    
    def get_distribution(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        return Normal(mean, std)
    
    def get_action_log_prob(self, x, action=None):
        dist = self.get_distribution(x)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def compute_returns_and_advantages(self, last_value=0):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [1])
        
        # Compute GAE (Generalized Advantage Estimation)
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = np.array(advantages) + np.array(self.values)
        advantages = np.array(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

def get_action(policy, state, deterministic=False):
    """Get action from policy - for visualization"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = policy.get_distribution(state_tensor)
        if deterministic:
            return dist.mean[0].cpu().numpy()
        return dist.sample()[0].cpu().numpy()

def load_best_model(actor, critic, model_path='best_humanoid_ppo.pth'):
    """Load the best model if it exists"""
    if os.path.exists(model_path):
        print(f"\n📂 Found existing model at {model_path}")
        checkpoint = torch.load(model_path)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        print("✅ Model loaded successfully!")
        return True
    else:
        print("\n📂 No existing model found. Starting fresh training.")
        return False


def train_humanoid_ppo():
    # Initialize environment
    env = gym.make('Humanoid-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("=" * 60)
    print("🚀 Starting Humanoid Training with PPO")
    print("=" * 60)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space low: {env.action_space.low[0]:.2f}")
    print(f"Action space high: {env.action_space.high[0]:.2f}")
    print("=" * 60)
    
    # Initialize networks
    actor = ActorNetwork(state_dim, action_dim, hidden_dim=512)
    critic = CriticNetwork(state_dim, hidden_dim=512)
    load_best_model(actor, critic)
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    
    # PPO hyperparameters
    num_episodes = 30
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    target_kl = 0.01
    update_epochs = 10
    mini_batch_size = 64
    horizon = 2048  # Steps to collect before update
    
    # Tracking variables
    best_reward = -np.inf
    reward_history = []
    running_reward = None
    episode_times = []
    buffer = PPOBuffer(gamma, gae_lambda)
    
    episode = 0
    total_steps = 0
    
    while episode < num_episodes:
        start_time = time.time()
        buffer.clear()
        
        # Collect trajectory
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(horizon):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                dist = actor.get_distribution(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(state_tensor)
            
            action_np = action[0].cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            buffer.store(state, action_np, reward, value.item(), log_prob.item(), done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if done:
                episode += 1
                episode_time = time.time() - start_time if episode == 1 else time.time() - start_time
                
                # Update running reward
                if running_reward is None:
                    running_reward = episode_reward
                else:
                    running_reward = 0.99 * running_reward + 0.01 * episode_reward
                
                reward_history.append(episode_reward)
                
                # Track best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save({
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                    }, 'best_humanoid_ppo.pth')
                    print(f"\n✨ New best reward: {best_reward:.2f} at episode {episode}")
                
                # Logging
                print(f"\n📊 Episode {episode:4d} | Reward: {episode_reward:7.2f} | "
                      f"Running: {running_reward:7.2f} | Best: {best_reward:7.2f} | "
                      f"Length: {episode_length:3d} | Steps: {total_steps}")
                
                # Reset for next episode
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                start_time = time.time()
                
                if episode >= num_episodes:
                    break
        
        # Compute last value for GAE
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            last_value = critic(state_tensor).item()
        
        # Compute returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(last_value)
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(buffer.states))
        actions = torch.FloatTensor(np.array(buffer.actions))
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO update epochs
        for epoch in range(update_epochs):
            # Mini-batch training
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), mini_batch_size):
                idx = indices[start_idx:start_idx + mini_batch_size]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Get current policy distributions
                dist = actor.get_distribution(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = critic(batch_states).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
                
                # Update actor
                actor_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizer.step()
                
                # Update critic
                critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()
            
            # KL divergence check for early stopping
            with torch.no_grad():
                dist = actor.get_distribution(states)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                kl = (old_log_probs - log_probs).mean().item()
                if kl > 1.5 * target_kl:
                    print(f"   Early stopping at epoch {epoch} due to KL={kl:.4f}")
                    break
        
        # Adaptive learning rate based on performance
        if episode > 0 and episode % 100 == 0:
            recent_rewards = reward_history[-100:]
            avg_recent = np.mean(recent_rewards)
            
            if avg_recent < 100:
                with torch.no_grad():
                    actor.log_std.data += 0.1
                print(f"   📈 Increasing exploration: log_std = {actor.log_std.mean().item():.2f}")
            
            print(f"\n📈 100-episode average: {avg_recent:.2f}")
        
        # Visualize every 200 episodes
        if episode > 0 and episode % 200 == 0:
            visualize_humanoid_ppo(actor, num_episodes=1)
    
    env.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Final running reward: {running_reward:.2f}")
    print("=" * 60)
    
    # Final visualization
    print("\n🎬 Final visualization of best policy...")
    visualize_humanoid_ppo(actor, num_episodes=3)

def visualize_humanoid_ppo(actor, num_episodes=2):
    """Visualize the trained policy in action"""
    print("\n" + "-" * 40)
    print("🎬 Visualizing trained Humanoid policy...")
    print("-" * 40)
    
    env = gym.make('Humanoid-v4', render_mode='human')
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:
            action = get_action(actor, state, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.02)
            
            if terminated or truncated:
                break
        
        print(f"   Test run {ep+1}: {total_reward:.2f} reward over {steps} steps")
        time.sleep(2)
    
    env.close()

def train_humanoid_with_curriculum_ppo():
    """Train with curriculum learning using PPO"""
    print("\n🎯 Starting Humanoid training with curriculum...")
    
    env = gym.make('Humanoid-v4', 
                   forward_reward_weight=1.0,
                   ctrl_cost_weight=0.1,
                   contact_cost_weight=0.1)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    
    difficulties = [0.5, 0.75, 1.0]
    
    for difficulty in difficulties:
        print(f"\n📈 Training at difficulty: {difficulty}")
        
        env = gym.make('Humanoid-v4',
                      forward_reward_weight=difficulty * 1.0,
                      ctrl_cost_weight=0.1 / difficulty,
                      contact_cost_weight=0.1 / difficulty)
        
        buffer = PPOBuffer()
        
        for episode in range(500):
            state, _ = env.reset()
            buffer.clear()
            episode_reward = 0
            
            for step in range(1000):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    dist = actor.get_distribution(state_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = critic(state_tensor)
                
                action_np = action[0].cpu().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                
                next_state, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                
                buffer.store(state, action_np, reward, value.item(), log_prob.item(), done)
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # PPO update (simplified for curriculum)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                last_value = critic(state_tensor).item()
            
            returns, advantages = buffer.compute_returns_and_advantages(last_value)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(buffer.states))
            actions = torch.FloatTensor(np.array(buffer.actions))
            old_log_probs = torch.FloatTensor(np.array(buffer.log_probs))
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
            
            # Single epoch update for curriculum
            dist = actor.get_distribution(states)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            values = critic(states).squeeze()
            value_loss = nn.MSELoss()(values, returns)
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            
            if (episode + 1) % 50 == 0:
                print(f"   Difficulty {difficulty}, Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    print("\n✅ Curriculum training complete!")
    return actor, critic

if __name__ == "__main__":
    # Standard PPO training
    train_humanoid_ppo()
    
    # Or use curriculum learning
    # actor, critic = train_humanoid_with_curriculum_ppo()
    # visualize_humanoid_ppo(actor, num_episodes=3)
