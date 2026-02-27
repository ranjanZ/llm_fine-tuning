import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.distributions import Normal
from collections import deque

# ================ NETWORK ARCHITECTURES ================

class ActorNetwork(nn.Module):
    """Policy network that outputs action distribution"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        return mean, self.log_std

class CriticNetwork(nn.Module):
    """Value network that estimates state value V(s)"""
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)

# ================ PPO CORE COMPONENTS ================

class PPOMemory:
    """Stores trajectories for PPO updates"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def get_tensors(self):
        """Convert memory to tensors"""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs, dtype=np.float32)),
            'rewards': torch.FloatTensor(np.array(self.rewards, dtype=np.float32)),
            'dones': torch.BoolTensor(np.array(self.dones)),
            'values': torch.FloatTensor(np.array(self.values, dtype=np.float32))
        }

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation
    A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
    where δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
    """
    advantages = []
    gae = 0
    
    # Convert to numpy for easier computation
    rewards = rewards.numpy()
    values = values.numpy()
    dones = dones.numpy()
    
    # Add dummy next value for last step
    values = np.append(values, 0)
    
    for t in reversed(range(len(rewards))):
        # If episode ended, next value is 0
        mask = 1 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
    
    # Compute returns (advantages + values)
    returns = [adv + values[t] for t, adv in enumerate(advantages)]
    
    return torch.FloatTensor(np.array(advantages)), torch.FloatTensor(np.array(returns))

# ================ PPO AGENT ================

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 lam=0.95, clip_epsilon=0.2, epochs=10, mini_batch_size=64,
                 max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01):
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.memory = PPOMemory()
        
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.actor(state)
            value = self.critic(state)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros(1)
            else:
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
        return action[0].cpu().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store transition in memory"""
        self.memory.store(state, action, log_prob, reward, done, value)
    
    def learn(self):
        """Update policy using PPO"""
        # Get all data as tensors
        data = self.memory.get_tensors()
        
        # Compute advantages and returns
        advantages, returns = compute_gae(
            data['rewards'], 
            data['values'], 
            data['dones'],
            self.gamma, 
            self.lam
        )
        
        # Normalize advantages (helps stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.epochs):
            # Create mini-batches
            indices = np.random.permutation(len(data['states']))
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                states = data['states'][batch_indices]
                actions = data['actions'][batch_indices]
                old_log_probs = data['log_probs'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                
                # Current policy outputs
                mean, log_std = self.actor(states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                
                # New log probs and entropy
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                           1 + self.clip_epsilon) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                values = self.critic(states).squeeze()
                critic_loss = nn.MSELoss()(values, returns_batch)
                
                # Total loss
                total_loss = (actor_loss + 
                            self.value_coef * critic_loss - 
                            self.entropy_coef * entropy)
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory after update
        self.memory.clear()
        
        n_updates = self.epochs * (len(data['states']) // self.mini_batch_size)
        return (total_actor_loss / n_updates, 
                total_critic_loss / n_updates, 
                total_entropy / n_updates)

# ================ TRAINING LOOP ================

def train_cheetah_ppo():
    # Initialize environment
    env = gym.make('HalfCheetah-v5')  # Use v5 instead of v4
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"🚀 Starting PPO Training on HalfCheetah")
    print("=" * 60)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print("=" * 60)
    
    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=64,
        entropy_coef=0.01
    )
    
    # Training parameters
    num_episodes = 2000
    max_steps = 1000
    update_frequency = 2048  # Collect this many steps before update
    print_every = 20
    
    # Tracking
    best_reward = -np.inf
    episode_rewards = deque(maxlen=100)
    step_count = 0
    episode_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Get action from agent
            action, log_prob, value = agent.get_action(state)
            
            # Clip action to environment bounds
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(
                state, action, log_prob, 
                reward, done, value
            )
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # Update if we have enough steps
            if step_count >= update_frequency:
                actor_loss, critic_loss, entropy = agent.learn()
                step_count = 0
                episode_count += 1
                
                if episode_count % 5 == 0:
                    print(f"   📈 Update {episode_count} - "
                          f"Actor: {actor_loss:.4f}, "
                          f"Critic: {critic_loss:.4f}, "
                          f"Entropy: {entropy:.4f}")
            
            if done:
                break
        
        # Track episode reward
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, 'best_ppo_cheetah.pth')
        
        # Logging
        if (episode + 1) % print_every == 0:
            print(f"\n📊 Episode {episode+1}")
            print(f"   Episode Reward: {episode_reward:.1f}")
            print(f"   Average Reward (100): {avg_reward:.1f}")
            print(f"   Best Reward: {best_reward:.1f}")
            print(f"   Episode Length: {episode_steps}")
            print(f"   Log Std: {agent.actor.log_std.mean().item():.2f}")
            print(f"   Value Range: [{min(agent.memory.values):.2f}, "
                  f"{max(agent.memory.values):.2f}]")
        
        # Visualize every 200 episodes
        if (episode + 1) % 200 == 0:
            visualize_cheetah_ppo(agent)
    
    env.close()
    print(f"\n✅ Training complete! Best reward: {best_reward:.1f}")

# ================ VISUALIZATION ================

def visualize_cheetah_ppo(agent, num_episodes=2):
    """Visualize the trained PPO agent"""
    print("\n🎬 Visualizing trained policy...")
    env = gym.make('HalfCheetah-v5', render_mode='human')
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:
            action, _, _ = agent.get_action(state, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.02)
            
            if terminated or truncated:
                break
        
        print(f"   Test run {ep+1}: {total_reward:.1f} reward over {steps} steps")
        time.sleep(1)
    
    env.close()

# ================ COMPARISON FUNCTION ================

def compare_with_reinforce():
    """Show the key differences between REINFORCE and PPO"""
    
    print("\n" + "="*60)
    print("🔍 PPO vs REINFORCE: Key Improvements")
    print("="*60)
    
    comparisons = [
        ("Update Mechanism", 
         "One update per episode", 
         "Multiple epochs with clipped objective"),
        
        ("Variance Reduction", 
         "Only return normalization", 
         "GAE + Advantage + Value baseline"),
        
        ("Sample Efficiency", 
         "Each step used once", 
         "Data reused for multiple epochs"),
        
        ("Stability", 
         "High variance, can collapse", 
         "Trust region via clipping"),
        
        ("Networks", 
         "Single policy network", 
         "Actor (policy) + Critic (value)"),
        
        ("Objective", 
         "max E[log π * R]", 
         "max E[min(ratio*A, clip(ratio)*A)]"),
    ]
    
    for feature, reinforce, ppo in comparisons:
        print(f"\n📌 {feature}:")
        print(f"   REINFORCE: {reinforce}")
        print(f"   PPO: {ppo}")

if __name__ == "__main__":
    print("\n🏃‍♂️ Converting REINFORCE to PPO for HalfCheetah")
    print("="*60)
    
    # Show comparison
    compare_with_reinforce()
    
    # Train with PPO
    train_cheetah_ppo()
