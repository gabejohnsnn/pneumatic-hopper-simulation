import numpy as np
from collections import deque

class HysteresisController:
    """
    Implements a hysteresis controller with built-in delay for the pneumatic hopper.
    This controls the altitude of the hopper by adjusting the thrust.
    """
    
    def __init__(self, target_height=3.0, hysteresis_band=0.2, max_thrust=1.0, 
                 min_thrust=0.0, response_delay=0.1, dt=0.01):
        """
        Initialize the hysteresis controller.
        
        Args:
            target_height (float): Target height for the hopper
            hysteresis_band (float): Width of the hysteresis band (dead zone)
            max_thrust (float): Maximum thrust output (normalized to 0-1)
            min_thrust (float): Minimum thrust output (normalized to 0-1)
            response_delay (float): Controller response delay in seconds
            dt (float): Time step in seconds
        """
        self.target_height = target_height
        self.hysteresis_band = hysteresis_band
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.response_delay = response_delay
        self.dt = dt
        
        # State variables
        self.current_output = 0.0
        
        # For delayed response
        self.delay_steps = int(response_delay / dt)
        self.error_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        self.height_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # For tracking history
        self.output_history = []
        self.error_history_log = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0
    
    def compute_control(self, current_height, estimated_velocity=None):
        """
        Compute the control output based on the current height and target height.
        Takes into account the hysteresis band and delay.
        
        Args:
            current_height (float): Current height of the hopper
            estimated_velocity (float, optional): Estimated velocity from Kalman filter
            
        Returns:
            float: Control output (normalized thrust between 0 and 1)
        """
        # Store current height in history
        self.height_history.append(current_height)
        
        # Calculate current error (target - actual)
        current_error = self.target_height - current_height
        self.error_history.append(current_error)
        
        # Get delayed error (accounts for pneumatic system delay)
        delayed_error = self.error_history[0]
        
        # Calculate half band width
        half_band = self.hysteresis_band / 2.0
        
        # Hysteresis logic with velocity-based enhancement if available
        if delayed_error > half_band:
            # Below target (outside lower band) - increase thrust
            self.current_output = self.max_thrust
        elif delayed_error < -half_band:
            # Above target (outside upper band) - decrease thrust
            self.current_output = self.min_thrust
        else:
            # Inside hysteresis band - maintain current thrust
            # If velocity information is available, use it to improve stability
            if estimated_velocity is not None:
                # If moving away from target while in the band, apply corrective thrust
                if delayed_error > 0 and estimated_velocity < -0.05:  # Moving down while below target
                    self.current_output = self.max_thrust
                elif delayed_error < 0 and estimated_velocity > 0.05:  # Moving up while above target
                    self.current_output = self.min_thrust
        
        # Update histories
        self.simulation_time += self.dt
        self.output_history.append(self.current_output)
        self.error_history_log.append(current_error)
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
        self.error_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        self.height_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        self.output_history = []
        self.error_history_log = []
        self.target_history = []
        self.time_history = []
        self.simulation_time = 0.0


# Test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test the hysteresis controller with a simple simulation
    dt = 0.01
    simulation_time = 10.0  # seconds
    steps = int(simulation_time / dt)
    
    # Initialize controller
    controller = HysteresisController(target_height=3.0, hysteresis_band=0.2, dt=dt)
    
    # Simple model for testing
    position = 2.0  # Start below target
    velocity = 0.0
    
    # Arrays to store results
    positions = np.zeros(steps)
    velocities = np.zeros(steps)
    control_outputs = np.zeros(steps)
    targets = np.zeros(steps)
    
    # Simple simulation
    for i in range(steps):
        # Change target halfway through
        if i == steps // 2:
            controller.set_target_height(4.0)
            
        # Get control output
        control = controller.compute_control(position, velocity)
        control_outputs[i] = control
        
        # Simple physics (thrust, gravity, damping)
        thrust_force = control * 20.0  # Max thrust 20N
        gravity_force = 9.81
        damping_force = 0.1 * velocity**2 * np.sign(-velocity)
        
        # Calculate acceleration
        acceleration = thrust_force - gravity_force + damping_force
        
        # Update velocity and position
        velocity += acceleration * dt
        position += velocity * dt
        
        # Record data
        positions[i] = position
        velocities[i] = velocity
        targets[i] = controller.get_target_height()
    
    # Plotting
    time = np.arange(0, simulation_time, dt)
    
    plt.figure(figsize=(12, 10))
    
    # Position and target
    plt.subplot(3, 1, 1)
    plt.plot(time, positions, 'b-', label='Position')
    plt.plot(time, targets, 'r--', label='Target')
    plt.ylabel('Position (m)')
    plt.grid(True)
    plt.legend()
    
    # Velocity
    plt.subplot(3, 1, 2)
    plt.plot(time, velocities, 'g-', label='Velocity')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    
    # Control output
    plt.subplot(3, 1, 3)
    plt.plot(time, control_outputs, 'k-', label='Control Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust Control (0-1)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('controller_test.png')
    plt.close()
    
    print("Test complete. Results saved to controller_test.png")
