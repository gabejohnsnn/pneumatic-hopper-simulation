"""
Controllers package for the pneumatic hopper simulation.
Includes different control methodologies.
"""

from .hysteresis_controller import HysteresisController
from .pid_controller import PIDController
from .bang_bang_controller import BangBangController
from .ddpg_controller import DDPGController
from .mpc_controller import MPCController
from .ppo_controller import PPOController

# Function to create a controller based on method name
def create_controller(method, target_height=3.0, **kwargs):
    """
    Factory function to create a controller of the specified type.
    
    Args:
        method (str): Controller type ("Hysteresis", "PID", "Bang-Bang", "DDPG", "MPC", "PPO")
        target_height (float): Target height for the controller
        **kwargs: Additional parameters for the specific controller
        
    Returns:
        Controller instance
    """
    if method == "Hysteresis":
        hysteresis_band = kwargs.get('band', 0.3)
        response_delay = kwargs.get('response_delay', 0.1)
        dt = kwargs.get('dt', 0.01)
        
        return HysteresisController(
            target_height=target_height,
            hysteresis_band=hysteresis_band,
            response_delay=response_delay,
            dt=dt
        )
    
    elif method == "PID":
        kp = kwargs.get('kp', 1.0)
        ki = kwargs.get('ki', 0.1)
        kd = kwargs.get('kd', 0.5)
        dt = kwargs.get('dt', 0.01)
        
        return PIDController(
            target_height=target_height,
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt
        )
    
    elif method == "Bang-Bang":
        threshold = kwargs.get('threshold', 0.1)
        dt = kwargs.get('dt', 0.01)
        
        return BangBangController(
            target_height=target_height,
            threshold=threshold,
            dt=dt
        )
    
    elif method == "DDPG":
        dt = kwargs.get('dt', 0.01)
        hidden_dim = kwargs.get('hidden_dim', 64)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        gamma = kwargs.get('gamma', 0.99)
        tau = kwargs.get('tau', 1e-3)
        noise_sigma = kwargs.get('noise_sigma', 0.2)
        pretrain = kwargs.get('pretrain', True)
        
        return DDPGController(
            target_height=target_height,
            dt=dt,
            hidden_dim=hidden_dim,
            actor_lr=learning_rate,
            critic_lr=learning_rate * 10,
            gamma=gamma,
            tau=tau,
            noise_sigma=noise_sigma,
            pretrain_nn=pretrain
        )
    
    elif method == "MPC":
        prediction_horizon = kwargs.get('prediction_horizon', 15)
        dt = kwargs.get('dt', 0.01)
        delay_steps = int(kwargs.get('response_delay', 0.1) / dt)
        max_thrust = kwargs.get('max_thrust', 20.0)
        mass = kwargs.get('mass', 1.0)
        
        return MPCController(
            target_height=target_height,
            prediction_horizon=prediction_horizon,
            dt=dt,
            delay_steps=delay_steps,
            max_thrust=max_thrust,
            mass=mass
        )
    
    elif method == "PPO":
        dt = kwargs.get('dt', 0.01)
        model_path = kwargs.get('model_path', None)
        training_mode = kwargs.get('training_mode', False)
        
        return PPOController(
            target_height=target_height,
            dt=dt,
            model_path=model_path,
            training_mode=training_mode
        )
    
    else:
        raise ValueError(f"Unknown controller method: {method}")
