"""
Bang-Bang controller implementation for the pneumatic hopper.
This implements an optimal control approach based on Pontryagin's maximum principle.
"""

import numpy as np

class BangBangController:
    """
    Implements a Bang-Bang controller for the pneumatic hopper.
    This controls the altitude of the hopper by switching between minimum and maximum thrust.
    """
    
    def __init__(self, target_height=3.0, threshold=0.1, max_thrust=1.0, 
                 min_thrust=0.0, dt=0.01, velocity_threshold=0.1):
        """
        Initialize the Bang-Bang controller.
        
        Args:
            target_height (float): Target height for the hopper
            threshold (float): Error threshold for switching control
            max_thrust (float): Maximum thrust output (normalized to 0-1)
            min_thrust (float): Minimum thrust output (normalized to 0-1)
            dt (float): Time step in seconds
            velocity_threshold (float): Velocity threshold for anticipatory control
        """
        self.target_height = target_height
        self.threshold = threshold
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.dt = dt
        self.velocity_threshold = velocity_threshold
        
        # State variables
        self.current_output = 0.0
        
        # For tracking history
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
    
    def compute_control(self, current_height, estimated_velocity=None):
        """
        Compute the Bang-Bang control output based on the current height and target height.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float, optional): Estimated velocity from Kalman filter
            
        Returns:
            float: Control output (normalized thrust between min_thrust and max_thrust)
        """
        # Calculate current error (target - actual)
        current_error = self.target_height - current_height
        
        # Basic Bang-Bang control logic
        if estimated_velocity is None:
            # Without velocity information, use simple error-based control
            if current_error > self.threshold:
                # Apply maximum thrust when below target
                self.current_output = self.max_thrust
            elif current_error < -self.threshold:
                # Apply minimum thrust when above target
                self.current_output = self.min_thrust
            # else: Keep current output in the dead zone
        else:
            # With velocity information, implement a more sophisticated switching curve
            # This approach is based on time-optimal control principles
            
            # Calculate the "switching curve" value
            # This is a simplified version of the optimal control policy
            # where we switch based on position and velocity
            
            # Estimate the distance needed to stop with minimum thrust
            # For upward motion, we need to consider how much height will be gained before stopping
            if estimated_velocity > 0:
                # Moving upward - consider how much height will be gained before stopping
                stopping_height = estimated_velocity**2 / (2 * 9.81)  # Simple v²/2a formula
                
                # If we're below target but will overshoot, or we're above target, apply minimum thrust
                if (current_error < 0) or (current_error > 0 and stopping_height > current_error):
                    self.current_output = self.min_thrust
                else:
                    self.current_output = self.max_thrust
            else:
                # Moving downward - consider how much height will be lost before stopping
                stopping_height = estimated_velocity**2 / (2 * 9.81)  # Simple v²/2a formula
                
                # If we're above target but will undershoot, or we're below target, apply maximum thrust
                if (current_error > 0) or (current_error < 0 and stopping_height > -current_error):
                    self.current_output = self.max_thrust
                else:
                    self.current_output = self.min_thrust
        
        # Update histories
        self.simulation_time += self.dt
        self.output_history.append(self.current_output)
        self.error_history.append(current_error)
        self.target_history.append(self.target_height)
        self.time_history.append(self.simulation_time)
        
        return self.current_output
    
    def set_target_height(self, height):
        """
        Set a new target height.
        
        Args:
            height (float): New target height
        """
        self.target_height = height
    
    def get_target_height(self):
        """
        Get the current target height.
        
        Returns:
            float: Target height
        """
        return self.target_height
    
    def adjust_target_height(self, delta):
        """
        Adjust the target height by a delta.
        
        Args:
            delta (float): Change in target height
        """
        self.target_height += delta
    
    def set_threshold(self, threshold):
        """
        Set a new threshold value.
        
        Args:
            threshold (float): New threshold value
        """
        self.threshold = threshold
    
    def get_history(self):
        """
        Get the history of control outputs, errors, and target heights.
        
        Returns:
            dict: Dictionary containing history data
        """
        return {
            'output': self.output_history,
            'error': self.error_history,
            'target': self.target_history,
            'time': self.time_history
        }
    
    def reset(self, target_height=None):
        """
        Reset the controller to initial state.
        
        Args:
            target_height (float, optional): New target height
        """
        if target_height is not None:
            self.target_height = target_height
            
        self.current_output = 0.0
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
