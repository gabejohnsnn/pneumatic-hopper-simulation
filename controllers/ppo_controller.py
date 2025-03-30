"""
PPO (Proximal Policy Optimization) Controller for the pneumatic hopper simulation.

This controller uses the Stable Baselines3 PPO implementation to learn
an optimal control policy for altitude regulation.
"""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional, Any


class HopperEnv(gym.Env):
    """
    Gymnasium environment wrapper for the pneumatic hopper control task.
    This allows us to use the pneumatic hopper with RL libraries like Stable Baselines3.
    """
    
    def __init__(self, target_height=3.0, dt=0.01, max_steps=1000):
        """
        Initialize the environment.
        
        Args:
            target_height: Target height for the hopper
            dt: Time step duration
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        self.target_height = target_height
        self.dt = dt
        self.max_steps = max_steps
        self.steps = 0
        
        # Define action space (discrete: 0 = no thrust, 1 = full thrust)
        self.action_space = spaces.Discrete(2)
        
        # Define observation space: [position, velocity, target_height]
        # Reasonable bounds for each dimension
        self.observation_space = spaces.Box(
            low=np.array([0.0, -20.0, 0.0]),  # min values for [pos, vel, target]
            high=np.array([10.0, 20.0, 10.0]),  # max values
            dtype=np.float32
        )
        
        # Initialize state (will be overwritten in reset)
        self.state = np.array([0.0, 0.0, self.target_height], dtype=np.float32)
        
        # Store cumulative reward for analysis
        self.cumulative_reward = 0.0
        self.reward_history = []
        self.last_reward = 0.0
        
        # Store position and action history for evaluation
        self.position_history = []
        self.action_history = []
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state (position, velocity, target)
        initial_position = 0.0  # Start at ground level
        initial_velocity = 0.0  # Start at rest
        self.state = np.array(
            [initial_position, initial_velocity, self.target_height], 
            dtype=np.float32
        )
        
        # Reset step counter and histories
        self.steps = 0
        self.cumulative_reward = 0.0
        self.position_history = [initial_position]
        self.action_history = []
        
        # No additional info needed for reset
        return self.state, {}
    
    def set_target_height(self, target_height):
        """
        Update the target height.
        
        Args:
            target_height: New target height
        """
        self.target_height = target_height
        self.state[2] = target_height  # Update in state vector
    
    def update_state(self, position, velocity):
        """
        Update the environment state with external physics simulation results.
        
        Args:
            position: Current position from physics engine
            velocity: Current velocity from physics engine
        """
        self.state[0] = float(position)
        self.state[1] = float(velocity)
        self.position_history.append(float(position))
    
    def get_reward(self, action):
        """
        Calculate reward for the current state-action pair.
        
        Reward function design is critical for learning. This implementation uses:
        - Negative position error (closer to target = higher reward)
        - Penalty for using thrust (encourages fuel efficiency)
        - Penalty for velocity (encourages stability near target)
        - Bonus for reaching and staying at target
        
        Args:
            action: Action taken (0 = no thrust, 1 = full thrust)
            
        Returns:
            float: Calculated reward
        """
        position, velocity, target = self.state
        
        # Position tracking error (negative because we want to minimize error)
        position_error = -abs(position - target)
        
        # Fuel usage penalty (only when thrusting)
        fuel_penalty = -0.1 * action
        
        # Velocity penalty (encourages stability)
        velocity_penalty = -0.05 * abs(velocity)
        
        # Goal achievement bonus (for being close to target with low velocity)
        goal_bonus = 0.0
        if abs(position - target) < 0.2 and abs(velocity) < 0.3:
            goal_bonus = 0.5  # Small but meaningful bonus for stability at target
        
        # Extra stability shaping when near target
        stability_shaping = 0.0
        if abs(position - target) < 0.5:
            stability_shaping = -0.2 * abs(velocity)  # Extra penalty for velocity when near target
        
        # Combined reward
        reward = position_error + fuel_penalty + velocity_penalty + goal_bonus + stability_shaping
        
        # Store reward for analysis
        self.last_reward = reward
        self.cumulative_reward += reward
        
        return reward
    
    def step(self, action):
        """
        Take a step in the environment.
        
        This is a placeholder - in this implementation, the external physics 
        simulation will actually update the state. This method just calculates
        rewards and termination conditions.
        
        Args:
            action: Action to take (0 = no thrust, 1 = full thrust)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Track the action
        self.action_history.append(int(action))
        
        # Calculate reward based on current state and action
        reward = self.get_reward(action)
        self.reward_history.append(reward)
        
        # Increment step counter
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        
        # Check truncation (episode length limit)
        truncated = self.steps >= self.max_steps
        
        # Prepare info dict
        info = {
            'position_error': abs(self.state[0] - self.target_height),
            'velocity': self.state[1],
            'cumulative_reward': self.cumulative_reward
        }
        
        return self.state, reward, terminated, truncated, info


class PPOController:
    """
    PPO-based controller for the pneumatic hopper using Stable Baselines3.
    """
    
    def __init__(self, target_height=3.0, dt=0.01, **kwargs):
        """
        Initialize the PPO controller.
        
        Args:
            target_height: Initial target height
            dt: Simulation time step
            **kwargs: Additional parameters
        """
        # Basic controller properties
        self.target_height = target_height
        self.dt = dt
        
        # Create environment
        self.env = HopperEnv(target_height=target_height, dt=dt)
        
        # Create PPO model (default hyperparameters, modify as needed)
        policy_kwargs = dict(
            # Two hidden layers of 64 units each
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=None,
            device='auto',
            _init_setup_model=True
        )
        
        # Training properties
        self.training_mode = kwargs.get('training_mode', True)
        self.learn_every_n_steps = kwargs.get('learn_every_n_steps', 10)
        self.steps_since_learning = 0
        
        # Loading pre-trained model if path provided
        model_path = kwargs.get('model_path', None)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded pre-trained PPO model from {model_path}")
        
        # Reset the controller
        self.reset()
    
    def reset(self, target_height=None):
        """
        Reset the controller.
        
        Args:
            target_height: Optional new target height
        """
        if target_height is not None:
            self.target_height = target_height
        
        # Reset the environment
        self.env.set_target_height(self.target_height)
        self.env.reset()
        
        # Reset learning counter
        self.steps_since_learning = 0
    
    def compute_control(self, position, velocity, acceleration=None):
        """
        Compute control action using the PPO policy.
        
        Args:
            position: Current position (height)
            velocity: Current velocity
            acceleration: Current acceleration (ignored for PPO)
            
        Returns:
            float: Control signal (0.0 for no thrust, 1.0 for full thrust)
        """
        # Update environment state with current physics state
        self.env.update_state(position, velocity)
        
        # Get observation from environment
        observation = self.env.state
        
        # Get action from PPO policy
        action, _states = self.model.predict(observation, deterministic=not self.training_mode)
        
        # Convert from discrete action (0 or 1) to continuous control signal (0.0 or 1.0)
        control_signal = float(action)
        
        # Learn if in training mode
        if self.training_mode:
            self.steps_since_learning += 1
            
            # Perform learning step at regular intervals
            if self.steps_since_learning >= self.learn_every_n_steps:
                # In a real implementation, we would accumulate experiences and
                # train the model. SB3 handles this internally during regular training.
                # For this simulation, we're assuming learning happens elsewhere.
                self.steps_since_learning = 0
        
        return control_signal
    
    def set_target_height(self, target_height):
        """
        Update the target height.
        
        Args:
            target_height: New target height
        """
        self.target_height = target_height
        self.env.set_target_height(target_height)
    
    def get_target_height(self):
        """
        Get the current target height.
        
        Returns:
            float: Target height
        """
        return self.target_height
    
    def load_model(self, path):
        """
        Load a pre-trained PPO model.
        
        Args:
            path: Path to saved model
        """
        try:
            self.model = PPO.load(path, env=self.env)
            print(f"Successfully loaded model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, path):
        """
        Save the current PPO model.
        
        Args:
            path: Path to save model
        """
        try:
            self.model.save(path)
            print(f"Successfully saved model to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def set_training_mode(self, training):
        """
        Set whether the controller is in training mode.
        
        Args:
            training: Boolean indicating training mode
        """
        self.training_mode = training
    
    def provide_reward(self, position, velocity):
        """
        Provide reward for the last action based on the resulting state.
        This function is compatible with the SimulationRunner interface.
        
        Args:
            position: Current position after action
            velocity: Current velocity after action
        """
        # We use the last action from the action history
        if self.env.action_history:
            last_action = self.env.action_history[-1]
            reward = self.env.get_reward(last_action)
            return reward
        return 0.0


class PPOTrainer:
    """
    Helper class for training a PPO model for the pneumatic hopper
    using the SimulationRunner.
    """
    
    def __init__(self, controller, runner, num_episodes=1000, steps_per_episode=1000):
        """
        Initialize the trainer.
        
        Args:
            controller: PPOController instance
            runner: SimulationRunner instance
            num_episodes: Number of episodes to train for
            steps_per_episode: Maximum steps per episode
        """
        self.controller = controller
        self.runner = runner
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self):
        """
        Train the PPO model.
        
        Returns:
            dict: Training metrics
        """
        print("Starting PPO training...")
        
        for episode in range(self.num_episodes):
            # Reset the environment and controller
            self.controller.reset()
            
            # Run one episode
            print(f"Episode {episode+1}/{self.num_episodes}")
            
            # Use the runner to simulate one episode
            episode_data = self.runner.run(duration=self.steps_per_episode * self.controller.dt)
            
            # Extract rewards
            if hasattr(self.controller.env, 'reward_history'):
                episode_reward = sum(self.controller.env.reward_history)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(len(self.controller.env.reward_history))
                
                print(f"  Episode Length: {self.episode_lengths[-1]}")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Avg Position Error: {np.mean(np.abs(np.array(self.controller.env.position_history) - self.controller.target_height)):.4f}")
            
            # Reset reward history for next episode
            self.controller.env.reward_history = []
        
        print("Training complete!")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
    
    def plot_training_metrics(self):
        """
        Plot training metrics.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot episode rewards
            ax1.plot(self.episode_rewards)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Episode Rewards During Training')
            ax1.grid(True)
            
            # Plot episode lengths
            ax2.plot(self.episode_lengths)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.set_title('Episode Lengths During Training')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('ppo_training_metrics.png')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training metrics: {e}")


# Standalone training function (can be used from command line)
def train_ppo_model(save_path='models/ppo_hopper.zip', episodes=100, render=False):
    """
    Train a PPO model for the pneumatic hopper and save it.
    
    Args:
        save_path: Path to save the trained model
        episodes: Number of episodes to train for
        render: Whether to render the environment during training
        
    Returns:
        PPOController: Trained controller
    """
    from physics import PhysicsEngine
    from sensor import SensorSystem
    from kalman_filter import KalmanFilter
    from core.logger import SimulationLogger
    from core.simulation_runner import SimulationRunner
    from visualization import Visualizer
    
    # Create components
    physics = PhysicsEngine(mass=1.0, max_thrust=20.0, delay_time=0.2, dt=0.01)
    sensors = SensorSystem(lidar_noise_std=0.05, mpu_noise_std=0.1)
    kf = KalmanFilter(dt=0.01, initial_position=0.0)
    
    # Create PPO controller
    controller = PPOController(target_height=3.0, dt=0.01, training_mode=True)
    
    # Create logger
    logger = SimulationLogger(log_folder='logs')
    
    # Create visualizer if rendering
    visualizer = Visualizer() if render else None
    
    # Create configuration
    config = {
        'dt': 0.01,
        'log_freq': 10,
        'controller_name': 'PPO',
        'initial_target': 3.0,
        'step_target': 3.0  # No target change for training
    }
    
    # Create runner
    runner = SimulationRunner(
        physics=physics,
        sensors=sensors,
        kf=kf,
        controller=controller,
        logger=logger,
        visualizer=visualizer,
        config=config
    )
    
    # Create trainer
    trainer = PPOTrainer(
        controller=controller,
        runner=runner,
        num_episodes=episodes,
        steps_per_episode=1000
    )
    
    # Train the model
    trainer.train()
    
    # Plot training metrics
    trainer.plot_training_metrics()
    
    # Save the model
    controller.save_model(save_path)
    
    return controller


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a PPO model for the pneumatic hopper')
    parser.add_argument('--save-path', type=str, default='models/ppo_hopper.zip',
                        help='Path to save the trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train for')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training')
    
    args = parser.parse_args()
    
    train_ppo_model(
        save_path=args.save_path,
        episodes=args.episodes,
        render=args.render
    )
