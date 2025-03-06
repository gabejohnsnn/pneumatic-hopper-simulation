"""
DDPG (Deep Deterministic Policy Gradient) reinforcement learning controller for the pneumatic hopper.

This implements a model-free reinforcement learning approach to control the altitude
of the hopper by learning optimal thrust policies through experience.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

# Define experience replay memory structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample transition experiences."""
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network that maps states to actions."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, init_w=3e-3):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Hidden layer dimension
            init_w (float): Weight initialization range
        """
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize final layer weights near zero for stable learning
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # Output is between 0 and 1 (normalized thrust)
        return torch.sigmoid(self.linear3(x))

class Critic(nn.Module):
    """Critic network that maps (state, action) pairs to Q-values."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, init_w=3e-3):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Hidden layer dimension
            init_w (float): Weight initialization range
        """
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Initialize final layer weights near zero for stable learning
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        """Forward pass through the network."""
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1):
        """
        Initialize the noise process.
        
        Args:
            size (int): Size of the action space
            mu (float): Mean of the noise
            theta (float): Rate of mean reversion
            sigma (float): Scale of the noise
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state."""
        self.state = self.mu.copy()
    
    def sample(self):
        """Generate a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class DDPGController:
    """
    Implements a DDPG controller with built-in learning for the pneumatic hopper.
    This learns to control the altitude of the hopper by optimizing thrust.
    """
    
    def __init__(self, target_height=3.0, state_dim=3, action_dim=1, hidden_dim=64,
                 gamma=0.99, tau=1e-3, actor_lr=1e-4, critic_lr=1e-3, 
                 buffer_size=10000, batch_size=64, update_every=4, dt=0.01,
                 noise_theta=0.15, noise_sigma=0.2, exploration_decay=0.9998,
                 min_exploration=0.05, pretrain_nn=True):
        """
        Initialize the DDPG controller.
        
        Args:
            target_height (float): Target height for the hopper
            state_dim (int): Dimension of the state space [position_error, velocity, acceleration]
            action_dim (int): Dimension of the action space [thrust]
            hidden_dim (int): Hidden layer dimension for NN
            gamma (float): Discount factor for future rewards
            tau (float): Soft update parameter for target networks
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            buffer_size (int): Capacity of the replay buffer
            batch_size (int): Batch size for training
            update_every (int): Number of steps between network updates
            dt (float): Time step in seconds
            noise_theta (float): OU noise mean reversion parameter
            noise_sigma (float): OU noise scale parameter
            exploration_decay (float): Factor to decay exploration noise over time
            min_exploration (float): Minimum exploration noise scale
            pretrain_nn (bool): Whether to initialize networks with a heuristic policy
        """
        self.target_height = target_height
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.dt = dt
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Device configuration (CPU or GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize target networks with the same weights
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Exploration noise
        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)
        self.noise_scale = 1.0
        
        # Learning step counter
        self.t_step = 0
        
        # Pretrain networks with a simple heuristic policy if requested
        if pretrain_nn:
            self._pretrain_networks()
        
        # State tracking
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.current_output = 0.0
        
        # For tracking history
        self.output_history = []
        self.error_history = []
        self.reward_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
        
        # Training mode
        self.training_mode = True
    
    def _hard_update(self, source, target):
        """Hard update: target = source"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def _soft_update(self, source, target):
        """Soft update: target = tau*source + (1-tau)*target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def _pretrain_networks(self):
        """Pretrain networks with a simple heuristic policy."""
        print("Pretraining DDPG networks with heuristic policy...")
        
        # Create synthetic experiences based on a simple control policy
        for _ in range(1000):
            # Generate random states around the target
            pos_error = np.random.uniform(-2.0, 2.0, size=(1, 1))
            velocity = np.random.uniform(-1.0, 1.0, size=(1, 1))
            acceleration = np.random.uniform(-12.0, 0.0, size=(1, 1))
            
            state = np.hstack([pos_error, velocity, acceleration])
            
            # Simple heuristic policy: apply thrust when below target and moving down
            if pos_error > 0 and velocity < 0:
                action = np.array([[0.8]])  # High thrust
            elif pos_error > 0 and velocity > 0.5:
                action = np.array([[0.4]])  # Medium thrust to slow ascent
            elif pos_error < 0 and velocity > 0:
                action = np.array([[0.0]])  # No thrust when above target and moving up
            elif pos_error < 0 and velocity < -0.5:
                action = np.array([[0.5]])  # Medium thrust to slow descent
            else:
                action = np.array([[0.4]])  # Balanced thrust near equilibrium
            
            # Create reward based on distance to target height and velocity
            reward = -abs(pos_error) - 0.1 * abs(velocity)
            
            # Next state based on simple dynamics
            next_pos_error = pos_error + velocity * self.dt
            next_velocity = velocity + acceleration * self.dt
            next_acceleration = -9.81 + action[0, 0] * 20.0  # Simplified dynamics
            
            next_state = np.hstack([next_pos_error, next_velocity, np.array([[next_acceleration]])])
            done = False
            
            # Add to memory
            self.memory.add(state, action, reward, next_state, done)
        
        # Train on these experiences
        for _ in range(100):
            if len(self.memory) > self.batch_size:
                self._learn()
        
        print("Pretraining complete.")
    
    def _get_state(self, position, velocity, acceleration=None):
        """
        Convert raw measurements to internal state representation.
        
        State is represented as [position_error, velocity, acceleration] where
        position_error = target_height - position
        """
        position_error = self.target_height - position
        
        if acceleration is None:
            acceleration = -9.81  # Default to gravity if no measurement
        
        state = np.array([[position_error, velocity, acceleration]], dtype=np.float32)
        return state
    
    def compute_control(self, current_height, estimated_velocity, estimated_acceleration=None):
        """
        Compute the control output based on the current state.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float): Estimated velocity from Kalman filter
            estimated_acceleration (float, optional): Estimated acceleration from Kalman filter
            
        Returns:
            float: Control output (normalized thrust between 0 and 1)
        """
        # Create state vector
        state = self._get_state(current_height, estimated_velocity, estimated_acceleration)
        
        # Convert numpy state to torch tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Set networks to evaluation mode during action selection
        self.actor.eval()
        
        with torch.no_grad():
            # Get action from actor network
            action = self.actor(state_tensor).cpu().data.numpy()
        
        # Set networks back to training mode if in training
        if self.training_mode:
            self.actor.train()
            
            # Add exploration noise in training mode
            noise = self.noise.sample() * self.noise_scale
            action = np.clip(action + noise, 0, 1)
            
            # Decay exploration noise
            self.noise_scale = max(self.min_exploration, self.noise_scale * self.exploration_decay)
        
        self.current_output = float(action[0, 0])
        
        # Save state and action for learning step
        if self.training_mode:
            self.last_state = state
            self.last_action = action
        
        # Update histories
        self.simulation_time += self.dt
        self.output_history.append(self.current_output)
        self.error_history.append(self.target_height - current_height)
        self.target_history.append(self.target_height)
        self.time_history.append(self.simulation_time)
        
        return self.current_output
    
    def provide_reward(self, current_height, estimated_velocity):
        """
        Provide reward for the last action taken.
        This should be called after each physics step to enable learning.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float): Estimated velocity from Kalman filter
        """
        if not self.training_mode or self.last_state is None or self.last_action is None:
            return
        
        # Calculate position error
        position_error = self.target_height - current_height
        
        # Calculate reward (negative cost)
        # Penalize position error and high velocities
        reward = -(position_error**2 + 0.1 * estimated_velocity**2)
        
        # Additional penalty for being close to the ground (safety margin)
        if current_height < 0.5:
            reward -= 2.0 * (0.5 - current_height)**2
        
        # Get next state
        next_state = self._get_state(current_height, estimated_velocity)
        
        # Determine if episode is done (not applicable in continuous simulation)
        done = False
        
        # Store experience in replay buffer
        self.memory.add(self.last_state, self.last_action, reward, next_state, done)
        
        # Save reward for history
        self.reward_history.append(reward)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self._learn()
    
    def _learn(self):
        """Update actor and critic networks based on sampled experiences."""
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_targets_next = self.critic_target(next_states, next_actions)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
    
    def set_target_height(self, height):
        """Set a new target height."""
        self.target_height = height
    
    def get_target_height(self):
        """Get the current target height."""
        return self.target_height
    
    def adjust_target_height(self, delta):
        """Adjust the target height by a delta."""
        self.target_height += delta
        # Ensure target height is reasonable
        self.target_height = max(0.5, self.target_height)
    
    def set_training_mode(self, training):
        """Set whether the controller is in training mode."""
        self.training_mode = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def save_networks(self, path):
        """Save the trained networks to files."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
        print(f"Saved DDPG networks to {path}")
    
    def load_networks(self, path):
        """Load trained networks from files."""
        if not torch.cuda.is_available():
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
            
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        print(f"Loaded DDPG networks from {path}")
    
    def get_history(self):
        """Get the history of control outputs, errors, and target heights."""
        return {
            'output': self.output_history,
            'error': self.error_history,
            'reward': self.reward_history,
            'target': self.target_history,
            'time': self.time_history
        }
    
    def reset(self, target_height=None):
        """Reset the controller to initial state."""
        if target_height is not None:
            self.target_height = target_height
            
        self.current_output = 0.0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.noise.reset()
        
        self.output_history = []
        self.error_history = []
        self.reward_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
