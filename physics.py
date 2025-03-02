import numpy as np
from collections import deque
import time

class PhysicsEngine:
    """
    Physics engine for simulating a pneumatic hopper with delayed control input.
    
    This implements a system modeled by: dx(t)/dt = f(x(t), u(t-τ))
    where the delay τ represents the pneumatic delay in the system.
    """
    
    def __init__(self, mass=1.0, gravity=9.81, max_thrust=20.0, delay_time=0.2, 
                 air_resistance=0.1, ground_height=0, dt=0.01):
        """
        Initialize the physics engine with system parameters.
        
        Args:
            mass (float): Mass of the hopper in kg
            gravity (float): Gravitational acceleration in m/s²
            max_thrust (float): Maximum thrust force in N
            delay_time (float): Pneumatic delay time in seconds
            air_resistance (float): Air resistance coefficient
            ground_height (float): Height of the ground in m
            dt (float): Simulation time step in seconds
        """
        self.mass = mass
        self.gravity = gravity
        self.max_thrust = max_thrust
        self.delay_time = delay_time
        self.air_resistance = air_resistance
        self.ground_height = ground_height
        self.dt = dt
        
        # State variables
        self.position = 2.0  # Initial height above ground (m)
        self.velocity = 0.0  # Initial velocity (m/s)
        self.acceleration = -gravity  # Initial acceleration (m/s²)
        self.thrust = 0.0  # Current thrust (N)
        
        # Initialize control input history for delay implementation
        self.delay_steps = int(delay_time / dt)
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # For tracking history
        self.position_history = []
        self.velocity_history = []
        self.thrust_history = []
        self.time_history = []
        self.simulation_time = 0.0
        
    def apply_control(self, control_input):
        """
        Apply a control input (0 to 1) which will be delayed by the pneumatic system.
        
        Args:
            control_input (float): Control input between 0 (no thrust) and 1 (max thrust)
        """
        # Constrain control input between 0 and 1
        control_input = np.clip(control_input, 0.0, 1.0)
        
        # Add to history for delayed application
        self.control_history.append(control_input)
    
    def step(self):
        """
        Step the simulation forward by dt seconds.
        Returns the current state (position, velocity, acceleration).
        """
        # Get the delayed control input
        delayed_control = self.control_history[0]
        
        # Calculate thrust from delayed control input
        self.thrust = delayed_control * self.max_thrust
        
        # Calculate forces
        gravity_force = self.mass * self.gravity
        drag_force = self.air_resistance * self.velocity**2 * np.sign(-self.velocity)
        
        # Calculate net force and acceleration
        net_force = self.thrust - gravity_force + drag_force
        self.acceleration = net_force / self.mass
        
        # Update velocity using Euler integration
        self.velocity += self.acceleration * self.dt
        
        # Update position using Euler integration
        self.position += self.velocity * self.dt
        
        # Ground collision check
        if self.position <= self.ground_height:
            self.position = self.ground_height
            # Implement a simple bounce with energy loss
            if self.velocity < 0:
                self.velocity = -0.5 * self.velocity  # 50% energy loss on bounce
        
        # Update simulation time
        self.simulation_time += self.dt
        
        # Record history
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.thrust_history.append(self.thrust)
        self.time_history.append(self.simulation_time)
        
        # Return current state
        return self.position, self.velocity, self.acceleration
    
    def get_history(self):
        """Return the history of position, velocity, thrust, and time."""
        return {
            'position': self.position_history,
            'velocity': self.velocity_history,
            'thrust': self.thrust_history,
            'time': self.time_history
        }
    
    def reset(self):
        """Reset the simulation to initial conditions."""
        self.position = 2.0
        self.velocity = 0.0
        self.acceleration = -self.gravity
        self.thrust = 0.0
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        self.position_history = []
        self.velocity_history = []
        self.thrust_history = []
        self.time_history = []
        self.simulation_time = 0.0


# Test code
if __name__ == "__main__":
    # Simple test to verify the physics engine
    engine = PhysicsEngine()
    
    # Run for 5 seconds
    total_steps = int(5.0 / engine.dt)
    
    for i in range(total_steps):
        # Apply a simple control strategy for testing
        if i * engine.dt < 2.0:
            engine.apply_control(0.0)  # No thrust for 2 seconds
        else:
            engine.apply_control(0.8)  # 80% thrust after 2 seconds
        
        pos, vel, acc = engine.step()
        
        if i % 20 == 0:  # Print every 20 steps
            print(f"Time: {i*engine.dt:.2f}s, Position: {pos:.2f}m, Velocity: {vel:.2f}m/s")
    
    print("Test complete.")
