"""
PID controller implementation for the pneumatic hopper.
"""

import numpy as np
from collections import deque

class PIDController:
    """
    Implements a PID controller for the pneumatic hopper.
    This controls the altitude of the hopper by adjusting the thrust.
    """
    
    def __init__(self, target_height=3.0, kp=1.0, ki=0.1, kd=0.5,
                 max_thrust=1.0, min_thrust=0.0, dt=0.01, 
                 integral_windup_limit=1.0):
        """
        Initialize the PID controller.
        
        Args:
            target_height (float): Target height for the hopper
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            max_thrust (float): Maximum thrust output (normalized to 0-1)
            min_thrust (float): Minimum thrust output (normalized to 0-1)
            dt (float): Time step in seconds
            integral_windup_limit (float): Limit for integral windup prevention
        """
        self.target_height = target_height
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.dt = dt
        self.integral_windup_limit = integral_windup_limit
        
        # PID state variables
        self.previous_error = 0.0
        self.integral = 0.0
        self.current_output = 0.0
        
        # For derivative filtering (simple moving average)
        self.derivative_filter_size = 3
        self.error_history = deque([0.0] * self.derivative_filter_size, maxlen=self.derivative_filter_size)
        
        # For tracking history
        self.output_history = []
        self.error_history_log = []
        self.target_history = []
        self.time_history = []
        self.p_terms = []
        self.i_terms = []
        self.d_terms = []
        self.simulation_time = 0.0
    
    def compute_control(self, current_height, estimated_velocity=None):
        """
        Compute the PID control output based on the current height and target height.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float, optional): Estimated velocity from Kalman filter
                If provided, used for improved derivative calculation
            
        Returns:
            float: Control output (normalized thrust between 0 and 1)
        """
        # Calculate current error (target - actual)
        current_error = self.target_height - current_height
        
        # Store error in history for filtering
        self.error_history.append(current_error)
        
        # Calculate P term
        p_term = self.kp * current_error
        
        # Calculate I term
        self.integral += current_error * self.dt
        
        # Apply anti-windup to integral term
        self.integral = np.clip(self.integral, -self.integral_windup_limit, self.integral_windup_limit)
        i_term = self.ki * self.integral
        
        # Calculate D term (with option to use Kalman velocity)
        if estimated_velocity is not None:
            # If velocity estimate is available, use it (negative because error = target - actual)
            d_term = self.kd * (-estimated_velocity)
        else:
            # Otherwise, use filtered derivative calculation
            filtered_error = sum(self.error_history) / len(self.error_history)
            derivative = (filtered_error - self.previous_error) / self.dt
            d_term = self.kd * derivative
            self.previous_error = filtered_error
        
        # Calculate total control output
        output = p_term + i_term + d_term
        
        # Clamp output to thrust limits
        self.current_output = np.clip(output, self.min_thrust, self.max_thrust)
        
        # Update histories
        self.simulation_time += self.dt
        self.output_history.append(self.current_output)
        self.error_history_log.append(current_error)
        self.target_history.append(self.target_height)
        self.time_history.append(self.simulation_time)
        self.p_terms.append(p_term)
        self.i_terms.append(i_term)
        self.d_terms.append(d_term)
        
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
    
    def set_gains(self, kp=None, ki=None, kd=None):
        """
        Update the PID gains.
        
        Args:
            kp (float, optional): New proportional gain
            ki (float, optional): New integral gain
            kd (float, optional): New derivative gain
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
    
    def get_history(self):
        """
        Get the history of control outputs, errors, and target heights.
        
        Returns:
            dict: Dictionary containing history data
        """
        return {
            'output': self.output_history,
            'error': self.error_history_log,
            'target': self.target_history,
            'time': self.time_history,
            'p_terms': self.p_terms,
            'i_terms': self.i_terms,
            'd_terms': self.d_terms
        }
    
    def reset(self, target_height=None):
        """
        Reset the controller to initial state.
        
        Args:
            target_height (float, optional): New target height
        """
        if target_height is not None:
            self.target_height = target_height
            
        self.previous_error = 0.0
        self.integral = 0.0
        self.current_output = 0.0
        self.error_history = deque([0.0] * self.derivative_filter_size, maxlen=self.derivative_filter_size)
        self.output_history = []
        self.error_history_log = []
        self.target_history = []
        self.time_history = []
        self.p_terms = []
        self.i_terms = []
        self.d_terms = []
        self.simulation_time = 0.0
