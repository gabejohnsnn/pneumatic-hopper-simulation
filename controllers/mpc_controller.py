"""
Model Predictive Control (MPC) implementation for the pneumatic hopper.

This controller uses a model of the system to predict future behavior over a horizon
and optimizes control inputs to minimize a cost function while respecting constraints.
It naturally handles the pneumatic delay by incorporating it into the prediction.

Features:
- Multi-objective cost optimization (position tracking, efficiency, smoothness)
- Air consumption modeling and efficiency optimization
- Prediction-based control with receding horizon
- Handling of pneumatic delay
"""

import numpy as np
import scipy.optimize as optimize
from collections import deque


class MPCController:
    """
    Implements an MPC controller for the pneumatic hopper.
    This optimizes the altitude control by predicting future states and minimizing a cost function.
    """
    
    def __init__(self, target_height=3.0, prediction_horizon=15, dt=0.01, 
                 response_delay=0.2, max_thrust=20.0, mass=1.0, gravity=9.81, 
                 air_resistance=0.1, consider_air_consumption=True,
                 air_efficiency_weight=2.0, air_usage_weight=1.0):
        """
        Initialize the MPC controller.
        
        Args:
            target_height (float): Target height for the hopper
            prediction_horizon (int): Number of steps to predict into the future
            dt (float): Time step in seconds
            response_delay (float): Pneumatic response delay in seconds
            max_thrust (float): Maximum thrust force in N
            mass (float): Mass of the hopper in kg
            gravity (float): Gravitational acceleration in m/s²
            air_resistance (float): Air resistance coefficient
            consider_air_consumption (bool): Whether to consider air consumption in optimization
            air_efficiency_weight (float): Weight for air efficiency in cost function
            air_usage_weight (float): Weight for air usage minimization in cost function
        """
        self.target_height = target_height
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.delay_steps = int(response_delay / dt)
        self.max_thrust = max_thrust
        self.mass = mass
        self.gravity = gravity
        self.air_resistance = air_resistance
        
        # Air consumption parameters
        self.consider_air_consumption = consider_air_consumption
        self.air_efficiency_weight = air_efficiency_weight
        self.air_usage_weight = air_usage_weight
        self.air_efficiency = 1.0  # Initial efficiency (will be updated by physics)
        self.air_remaining = 100.0  # Initial air remaining percentage
        
        # Control bounds (thrust between 0-1)
        self.bounds = [(0, 1)] * prediction_horizon
        
        # Initial control guess (half thrust)
        self.initial_guess = np.ones(prediction_horizon) * 0.5
        
        # Control history for delay handling
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # For tracking history
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.air_efficiency_history = []
        self.air_remaining_history = []
        self.simulation_time = 0.0
        
        # Last optimization result for warm starting
        self.last_solution = None
    
    def predict_state(self, state, control, dt, air_efficiency=None):
        """
        Predict the next state given the current state and control input.
        
        Args:
            state (numpy.ndarray): Current state [position, velocity, acceleration]
            control (float): Normalized control input (0-1)
            dt (float): Time step
            air_efficiency (float, optional): Air efficiency factor (0-1)
            
        Returns:
            numpy.ndarray: Predicted next state
        """
        pos, vel, acc = state
        
        # Apply air efficiency if specified
        actual_control = control
        if air_efficiency is not None and self.consider_air_consumption:
            actual_control = control * air_efficiency
        
        # Calculate forces and acceleration
        thrust_force = actual_control * self.max_thrust
        gravity_force = self.mass * self.gravity
        drag_force = self.air_resistance * vel**2 * np.sign(-vel)
        
        # Calculate net force and new acceleration
        net_force = thrust_force - gravity_force + drag_force
        new_acc = net_force / self.mass
        
        # Update velocity and position using Euler integration
        new_vel = vel + acc * dt  # Use current acceleration for velocity update
        new_pos = pos + vel * dt + 0.5 * acc * dt**2  # Use current values for position
        
        # Handle ground collision
        if new_pos <= 0:
            new_pos = 0
            if new_vel < 0:
                new_vel = -0.5 * new_vel  # 50% energy loss on bounce
        
        return np.array([new_pos, new_vel, new_acc])
    
    def simulate_trajectory(self, initial_state, control_sequence):
        """
        Simulate the system trajectory over the prediction horizon.
        
        Args:
            initial_state (numpy.ndarray): Initial state [position, velocity, acceleration]
            control_sequence (numpy.ndarray): Sequence of control inputs
            
        Returns:
            tuple: (state_trajectory, air_usage_trajectory)
        """
        # Get control history for delay handling
        control_buffer = list(self.control_history)
        
        # Combine with new controls for full sequence
        full_control_sequence = control_buffer + list(control_sequence)
        
        # Initialize trajectory with initial state
        states = [initial_state]
        state = initial_state.copy()
        
        # For tracking air usage in prediction
        air_usage = []
        
        # Model air efficiency decay as air is used (simplified prediction)
        if self.consider_air_consumption:
            # Start with current air efficiency
            current_efficiency = self.air_efficiency
            
            # Estimate remaining percentage at each step
            remaining_percentage = self.air_remaining
        else:
            current_efficiency = None
        
        # Simulate system forward
        for i in range(self.prediction_horizon):
            # Apply delayed control (current step is at the end of the buffer)
            control = full_control_sequence[i]
            
            # Consider control as air usage
            air_usage.append(control)
            
            # Predict next state with possible efficiency impact
            state = self.predict_state(state, control, self.dt, current_efficiency)
            states.append(state)
            
            # Update predicted air efficiency for next state (simplified model)
            if self.consider_air_consumption:
                # Decay efficiency slightly for each control action (empirical prediction)
                # Could be refined with a better model based on the AirConsumptionModel
                air_usage_impact = control * 0.0005  # Empirical impact on efficiency
                current_efficiency = max(0.85, current_efficiency - air_usage_impact)
                
                # Estimate air usage impact on remaining percentage
                air_used = control * 0.01  # Rough estimate of percentage used per step
                remaining_percentage = max(0, remaining_percentage - air_used)
                
                # If air would run out in prediction, dramatically reduce efficiency
                if remaining_percentage < 10:
                    current_efficiency *= 0.9
                if remaining_percentage <= 0:
                    current_efficiency = 0.0
        
        return np.array(states), np.array(air_usage)
    
    def compute_cost(self, trajectory, control_sequence, air_usage=None):
        """
        Compute the cost for a given trajectory and control sequence.
        
        Args:
            trajectory (numpy.ndarray): Predicted state trajectory
            control_sequence (numpy.ndarray): Control input sequence
            air_usage (numpy.ndarray, optional): Air usage trajectory
            
        Returns:
            float: Cost value (lower is better)
        """
        # Extract positions from trajectory
        positions = trajectory[:, 0]
        velocities = trajectory[:, 1]
        
        # Position error cost (weighted heavily)
        position_error = np.sum((positions - self.target_height) ** 2) * 10.0
        
        # Velocity minimization (for smooth hovering)
        velocity_cost = np.sum(velocities ** 2) * 1.0
        
        # Control effort cost (minimize thrust usage)
        control_cost = np.sum(control_sequence ** 2) * 0.1
        
        # Control smoothness cost (penalize rapid changes)
        if len(control_sequence) > 1:
            smoothness_cost = np.sum(np.diff(control_sequence) ** 2) * 5.0
        else:
            smoothness_cost = 0.0
        
        # Terminal cost (be close to target at end of horizon)
        terminal_cost = ((positions[-1] - self.target_height) ** 2) * 50.0
        
        # Air efficiency and consumption cost (if enabled)
        air_efficiency_cost = 0.0
        air_usage_total_cost = 0.0
        
        if self.consider_air_consumption and air_usage is not None:
            # Higher cost for lower efficiency
            air_efficiency_factor = max(0.01, self.air_efficiency)  # Avoid division by zero
            air_efficiency_cost = (1.0 / air_efficiency_factor - 1.0) * self.air_efficiency_weight * 10.0
            
            # Penalize excessive air usage (weighted by current availability)
            remaining_factor = max(0.1, self.air_remaining / 100.0)  # Scale based on percentage
            air_usage_total_cost = np.sum(air_usage) * self.air_usage_weight * (1.0 / remaining_factor)
        
        # Total cost
        total_cost = (position_error + velocity_cost + control_cost + 
                     smoothness_cost + terminal_cost + 
                     air_efficiency_cost + air_usage_total_cost)
        
        return total_cost
    
    def optimize_control(self, initial_state):
        """
        Optimize the control sequence for the given initial state.
        
        Args:
            initial_state (numpy.ndarray): Current state [position, velocity, acceleration]
            
        Returns:
            numpy.ndarray: Optimized control sequence
        """
        # Use last solution as initial guess if available (warm start)
        if self.last_solution is not None:
            # Shift the solution and repeat the last value
            self.initial_guess = np.append(self.last_solution[1:], self.last_solution[-1])
        
        # Define objective function for optimization
        def objective(control_sequence):
            trajectory, air_usage = self.simulate_trajectory(initial_state, control_sequence)
            return self.compute_cost(trajectory, control_sequence, air_usage)
        
        # Use Sequential Least Squares Programming for optimization
        try:
            result = optimize.minimize(
                objective,
                self.initial_guess,
                method='SLSQP',
                bounds=self.bounds,
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success:
                self.last_solution = result.x
                return result.x
            else:
                print(f"MPC optimization failed: {result.message}")
                return self.initial_guess
                
        except Exception as e:
            print(f"Error in MPC optimization: {e}")
            return self.initial_guess
    
    def compute_control(self, current_height, estimated_velocity, estimated_acceleration=None):
        """
        Compute the control input based on the current state.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float): Estimated velocity from Kalman filter
            estimated_acceleration (float, optional): Estimated acceleration from Kalman filter
            
        Returns:
            float: Control output (normalized thrust between 0 and 1)
        """
        # Create state vector
        if estimated_acceleration is None:
            estimated_acceleration = -self.gravity
        
        state = np.array([current_height, estimated_velocity, estimated_acceleration])
        
        # Optimize control sequence
        control_sequence = self.optimize_control(state)
        
        # Get the first control action
        control = float(control_sequence[0])
        
        # Update control history
        self.control_history.append(control)
        
        # Update tracking history
        self.simulation_time += self.dt
        self.output_history.append(control)
        self.error_history.append(self.target_height - current_height)
        self.target_history.append(self.target_height)
        self.time_history.append(self.simulation_time)
        self.air_efficiency_history.append(self.air_efficiency)
        self.air_remaining_history.append(self.air_remaining)
        
        return control
    
    def update_air_status(self, efficiency, remaining_percentage):
        """
        Update the internal air consumption status to inform the optimizer.
        
        Args:
            efficiency (float): Current air efficiency (0-1)
            remaining_percentage (float): Percentage of air remaining (0-100)
        """
        self.air_efficiency = efficiency
        self.air_remaining = remaining_percentage
    
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
    
    def get_history(self):
        """Get the history of control outputs, errors, and target heights."""
        history = {
            'output': self.output_history,
            'error': self.error_history,
            'target': self.target_history,
            'time': self.time_history
        }
        
        # Include air consumption data if used
        if self.consider_air_consumption:
            history.update({
                'air_efficiency': self.air_efficiency_history,
                'air_remaining': self.air_remaining_history
            })
            
        return history
    
    def reset(self, target_height=None):
        """Reset the controller to initial state."""
        if target_height is not None:
            self.target_height = target_height
            
        # Reset control history
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # Reset optimization state
        self.initial_guess = np.ones(self.prediction_horizon) * 0.5
        self.last_solution = None
        
        # Reset air consumption status to initial values
        self.air_efficiency = 1.0
        self.air_remaining = 100.0
        
        # Reset tracking history
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.air_efficiency_history = []
        self.air_remaining_history = []
        self.simulation_time = 0.0

# Example usage (for testing)
if __name__ == "__main__":
    # Test the MPC controller with a simple example
    controller = MPCController(target_height=3.0, consider_air_consumption=True)
    
    # Initial state (at height 2.0)
    current_height = 2.0
    current_velocity = 0.0
    current_acceleration = -9.81
    
    # Initial air status
    controller.update_air_status(1.0, 100.0)
    
    # Compute control
    control = controller.compute_control(current_height, current_velocity, current_acceleration)
    print(f"Initial state: h={current_height}, v={current_velocity}, a={current_acceleration}")
    print(f"Computed control: {control}")
    
    # Simulate a few steps with diminishing air
    for i in range(10):
        # Update state (simple simulation)
        current_acceleration = control * 20.0 / 1.0 - 9.81  # thrust - gravity
        current_velocity += current_acceleration * 0.01
        current_height += current_velocity * 0.01
        
        # Simulate decreasing air efficiency to test adaptation
        simulated_efficiency = 1.0 - (i * 0.05)  # Decreasing efficiency
        simulated_remaining = 100.0 - (i * 5.0)  # Decreasing remaining air
        controller.update_air_status(simulated_efficiency, simulated_remaining)
        
        # Compute new control
        control = controller.compute_control(current_height, current_velocity, current_acceleration)
        
        print(f"Step {i+1}: h={current_height:.3f}, v={current_velocity:.3f}, control={control:.3f}, " +
              f"air_efficiency={simulated_efficiency:.2f}, air_remaining={simulated_remaining:.1f}%")
