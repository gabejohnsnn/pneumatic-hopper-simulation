"""
Model Predictive Control (MPC) implementation for the pneumatic hopper.

This controller uses a model of the system to predict future behavior over a horizon
and optimizes control inputs to minimize a cost function while respecting constraints.
It naturally handles the pneumatic delay by incorporating it into the prediction.
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
                 delay_steps=20, max_thrust=20.0, mass=1.0, gravity=9.81, 
                 air_resistance=0.1):
        """
        Initialize the MPC controller.
        
        Args:
            target_height (float): Target height for the hopper
            prediction_horizon (int): Number of steps to predict into the future
            dt (float): Time step in seconds
            delay_steps (int): Number of time steps in the pneumatic delay
            max_thrust (float): Maximum thrust force in N
            mass (float): Mass of the hopper in kg
            gravity (float): Gravitational acceleration in m/s²
            air_resistance (float): Air resistance coefficient
        """
        self.target_height = target_height
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.delay_steps = delay_steps
        self.max_thrust = max_thrust
        self.mass = mass
        self.gravity = gravity
        self.air_resistance = air_resistance
        
        # Control bounds (thrust between 0-1)
        self.bounds = [(0, 1)] * prediction_horizon
        
        # Initial control guess (half thrust)
        self.initial_guess = np.ones(prediction_horizon) * 0.5
        
        # Control history for delay handling
        self.control_history = deque([0.0] * delay_steps, maxlen=delay_steps)
        
        # For tracking history
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
        
        # Last optimization result for warm starting
        self.last_solution = None
    
    def predict_state(self, state, control, dt):
        """
        Predict the next state given the current state and control input.
        
        Args:
            state (numpy.ndarray): Current state [position, velocity, acceleration]
            control (float): Normalized control input (0-1)
            dt (float): Time step
            
        Returns:
            numpy.ndarray: Predicted next state
        """
        pos, vel, acc = state
        
        # Calculate forces and acceleration
        thrust_force = control * self.max_thrust
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
            numpy.ndarray: Predicted state trajectory
        """
        # Get control history for delay handling
        control_buffer = list(self.control_history)
        
        # Combine with new controls for full sequence
        full_control_sequence = control_buffer + list(control_sequence)
        
        # Initialize trajectory with initial state
        states = [initial_state]
        state = initial_state.copy()
        
        # Simulate system forward
        for i in range(self.prediction_horizon):
            # Apply delayed control (current step is at the end of the buffer)
            control = full_control_sequence[i]
            
            # Predict next state
            state = self.predict_state(state, control, self.dt)
            states.append(state)
        
        return np.array(states)
    
    def compute_cost(self, trajectory, control_sequence):
        """
        Compute the cost for a given trajectory and control sequence.
        
        Args:
            trajectory (numpy.ndarray): Predicted state trajectory
            control_sequence (numpy.ndarray): Control input sequence
            
        Returns:
            float: Cost value (lower is better)
        """
        # Extract positions from trajectory
        positions = trajectory[:, 0]
        velocities = trajectory[:, 1]
        
        # Position error cost (weighted more heavily)
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
        
        # Total cost
        total_cost = position_error + velocity_cost + control_cost + smoothness_cost + terminal_cost
        
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
            trajectory = self.simulate_trajectory(initial_state, control_sequence)
            return self.compute_cost(trajectory, control_sequence)
        
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
        
        return control
    
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
        return {
            'output': self.output_history,
            'error': self.error_history,
            'target': self.target_history,
            'time': self.time_history
        }
    
    def reset(self, target_height=None):
        """Reset the controller to initial state."""
        if target_height is not None:
            self.target_height = target_height
            
        # Reset control history
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # Reset optimization state
        self.initial_guess = np.ones(self.prediction_horizon) * 0.5
        self.last_solution = None
        
        # Reset tracking history
        self.output_history = []
        self.error_history = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0

# Example usage (for testing)
if __name__ == "__main__":
    # Test the MPC controller with a simple example
    controller = MPCController(target_height=3.0)
    
    # Initial state (at height 2.0)
    current_height = 2.0
    current_velocity = 0.0
    current_acceleration = -9.81
    
    # Compute control
    control = controller.compute_control(current_height, current_velocity, current_acceleration)
    print(f"Initial state: h={current_height}, v={current_velocity}, a={current_acceleration}")
    print(f"Computed control: {control}")
    
    # Simulate a few steps
    for i in range(10):
        # Update state (simple simulation)
        current_acceleration = control * 20.0 / 1.0 - 9.81  # thrust - gravity
        current_velocity += current_acceleration * 0.01
        current_height += current_velocity * 0.01
        
        # Compute new control
        control = controller.compute_control(current_height, current_velocity, current_acceleration)
        
        print(f"Step {i+1}: h={current_height:.3f}, v={current_velocity:.3f}, control={control:.3f}")
