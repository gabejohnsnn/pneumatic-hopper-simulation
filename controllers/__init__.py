"""
Controllers package for the pneumatic hopper simulation.
Includes different control methodologies.
"""

from .hysteresis_controller import HysteresisController
from .pid_controller import PIDController
from .bang_bang_controller import BangBangController

# Function to create a controller based on method name
def create_controller(method, target_height=3.0, **kwargs):
    """
    Factory function to create a controller of the specified type.
    
    Args:
        method (str): Controller type ("Hysteresis", "PID", "Bang-Bang")
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
    
    else:
        raise ValueError(f"Unknown controller method: {method}")
