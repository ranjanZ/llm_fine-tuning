import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.distributions import Normal
import wandb
from tqdm import tqdm
import os

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==================== NEURAL NETWORK ARCHITECTURE ====================

class ActorNetwork(nn.Module):
    """
    The Actor: Learns the policy π(a|s)
    For continuous control, we output mean and log_std of a Gaussian distribution
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        
        # Mean of action distribution
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
            
    def forward(self, state):
        features = self.feature_net(state)
        mean = self.mean_net(features)
        std = torch.exp(self.log_std.clamp(-20, 2))  # Clamp for stability
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy distribution
        """
        mean, std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)  # Actions are tanh-squashed to [-1, 1]
        
        # Create normal distribution and sample
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Apply tanh squashing and correct log probability
        action = torch.tanh(action)
        # Correction for tanh squashing: log_prob -= log(1 - tanh^2 + epsilon)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    The Critic: Estimates value function V(s)
    """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
            
    def forward(self, state):
        return self.net(state)


# ==================== BUFFER FOR EXPERIENCE COLLECTION ====================

class RolloutBuffer:
    """
    Stores trajectories for PPO updates
    """
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device
        self.clear()
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)).to(self.device),
            torch.FloatTensor(np.array(self.actions)).to(self.device),
            torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            torch.FloatTensor(np.array(self.rewards)).to(self.device),
            torch.FloatTensor(np.array(self.dones)).to(self.device),
            torch.FloatTensor(np.array(self.values)).to(self.device),
        )
    
    def __len__(self):
        return len(self.states)


# ==================== PPO ALGORITHM ====================

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        mini_batch_size=64,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize buffer
        self.buffer = None
        
    def get_action(self, state, deterministic=False):
        """
        Get action from policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action(state, deterministic=True)[0]
                return action.cpu().numpy()[0]
            
            action, log_prob = self.actor.get_action(state)
            value = self.critic(state)
            
        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0, 0]
        )
    
    def compute_gae(self, rewards, dones, values):
        """
        Compute Generalized Advantage Estimation
        """
        advantages = []
        gae = 0
        
        # Reverse iteration for GAE calculation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # No next state
                next_non_terminal = 1 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def update(self):
        """
        Update policy using PPO clipped objective
        """
        # Get data from buffer
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.update_epochs):
            # Generate mini-batches
            batch_size = len(states)
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Get current policy probabilities
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                
                # Compute log probabilities with tanh correction
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                log_probs -= torch.log(1 - batch_actions.pow(2) + 1e-6).sum(dim=-1)
                
                # Compute entropy for exploration
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Compute probability ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Subtract entropy bonus (we want to maximize entropy, so subtract in loss)
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                # Critic loss
                values_pred = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(batch_returns, values_pred)
                
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        n_updates = self.update_epochs * (batch_size // self.mini_batch_size)
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


# ==================== MAIN TRAINING LOOP ====================

def train_humanoid(config):
    """
    Main training function for Humanoid-v2
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create environment
    env = gym.make('Humanoid-v2')
    eval_env = gym.make('Humanoid-v2')  # Separate env for evaluation
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        entropy_coef=config['entropy_coef'],
        value_coef=config['value_coef'],
        max_grad_norm=config['max_grad_norm'],
        update_epochs=config['update_epochs'],
        mini_batch_size=config['mini_batch_size'],
        device=device
    )
    
    # Initialize buffer
    agent.buffer = RolloutBuffer(
        buffer_size=config['num_steps'],
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu'  # Store on CPU initially
    )
    
    # Initialize logging
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            config=config,
            name=f"Humanoid-PPO-seed{config['seed']}"
        )
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_eval_reward = -float('inf')
    
    # Main training loop
    state = env.reset()
    episode_reward = 0
    episode_step = 0
    total_steps = 0
    updates = 0
    
    progress_bar = tqdm(total=config['total_timesteps'], desc="Training")
    
    while total_steps < config['total_timesteps']:
        # Collect trajectory
        for step in range(config['num_steps']):
            # Get action from policy
            action, log_prob, value = agent.get_action(state)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.buffer.add(state, action, log_prob, reward, done, value)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_step += 1
            total_steps += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                state = env.reset()
                episode_reward = 0
                episode_step = 0
            
            progress_bar.update(1)
        
        # Update policy
        update_info = agent.update()
        updates += 1
        
        # Clear buffer
        agent.buffer.clear()
        
        # Evaluate periodically
        if updates % config['eval_frequency'] == 0:
            eval_reward = evaluate(agent, eval_env, num_episodes=5)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_model(agent, config['model_path'], eval_reward)
            
            # Log metrics
            if config['use_wandb']:
                wandb.log({
                    'train/avg_reward_100': np.mean(episode_rewards) if episode_rewards else 0,
                    'train/avg_length_100': np.mean(episode_lengths) if episode_lengths else 0,
                    'train/actor_loss': update_info['actor_loss'],
                    'train/critic_loss': update_info['critic_loss'],
                    'train/entropy': update_info['entropy'],
                    'eval/avg_reward': eval_reward,
                    'total_steps': total_steps,
                })
            else:
                print(f"\nUpdate {updates}, Steps {total_steps}:")
                print(f"  Train Reward (100 ep): {np.mean(episode_rewards):.2f}")
                print(f"  Eval Reward: {eval_reward:.2f}")
                print(f"  Actor Loss: {update_info['actor_loss']:.4f}")
                print(f"  Entropy: {update_info['entropy']:.4f}")
    
    progress_bar.close()
    env.close()
    eval_env.close()
    
    return agent


def evaluate(agent, env, num_episodes=5):
    """
    Evaluate agent performance
    """
    rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)


def save_model(agent, path, eval_reward):
    """
    Save model checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
        'eval_reward': eval_reward
    }, path)
    print(f"Model saved to {path}")


def load_model(path, agent):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
    agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
    return agent


# ==================== CONFIGURATION ====================

def get_config():
    """
    Get training configuration
    """
    return {
        # Environment
        'env_name': 'Humanoid-v2',
        'seed': 42,
        
        # Training
        'total_timesteps': 10_000_000,  # 10 million steps
        'num_steps': 2048,  # Steps per update
        'update_epochs': 10,
        'mini_batch_size': 64,
        
        # PPO Hyperparameters
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Evaluation
        'eval_frequency': 10,  # Evaluate every N updates
        'model_path': 'models/humanoid_ppo_best.pt',
        
        # Logging
        'use_wandb': False,  # Set to True if you want to use Weights & Biases
        'wandb_project': 'humanoid-ppo'
    }


# ==================== VISUALIZATION AND TESTING ====================

def visualize_trained_agent(model_path, num_episodes=5, render=True):
    """
    Visualize a trained agent
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env = gym.make('Humanoid-v2', render_mode='human' if render else None)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Load trained model
    agent = load_model(model_path, agent)
    agent.actor.eval()
    agent.critic.eval()
    
    # Run episodes
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get deterministic action
            action = agent.get_action(state, deterministic=True)
            
            # Step environment
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


# ==================== MAIN ====================

if __name__ == "__main__":
    # Get configuration
    config = get_config()
    
    # Train agent
    agent = train_humanoid(config)
    
    print("Training completed!")
    
    # Optional: Visualize the trained agent
    # Uncomment the line below to watch your trained agent walk!
    # visualize_trained_agent(config['model_path'], num_episodes=3, render=True)
