import numpy as np
from collections import deque
import time
from air_consumption import AirConsumptionModel

class PhysicsEngine:
    """
    Physics engine for simulating a pneumatic hopper with delayed control input.
    
    This implements a system modeled by: dx(t)/dt = f(x(t), u(t-τ))
    where the delay τ represents the pneumatic delay in the system.
    
    Features:
    - Realistic pneumatic delay modeling
    - Air resistance
    - Ground collision with energy loss
    - Advanced air consumption modeling
    """
    
    def __init__(self, mass=1.0, gravity=9.81, max_thrust=20.0, delay_time=0.2, 
                 air_resistance=0.1, ground_height=0, dt=0.01,
                 enable_air_consumption=True, tank_volume=0.02, initial_pressure=20e6):
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
            enable_air_consumption (bool): Enable advanced air consumption modeling
            tank_volume (float): Volume of compressed air tank in m³
            initial_pressure (float): Initial tank pressure in Pa
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
        
        # Air consumption modeling
        self.enable_air_consumption = enable_air_consumption
        if enable_air_consumption:
            self.air_model = AirConsumptionModel(
                tank_volume=tank_volume,
                initial_pressure=initial_pressure,
                max_thrust=max_thrust,
                dt=dt
            )
        else:
            self.air_model = None
        
        # Air consumption state variables
        self.air_flow_rate = 0.0  # Current mass flow rate (kg/s)
        self.air_efficiency = 1.0  # Current flow efficiency (0-1)
        self.air_remaining = 100.0  # Percentage of air remaining
        self.tank_empty = False  # Flag for empty tank
        
        # Variable mass support (as air is depleted from system)
        self.variable_mass = False  # Enable variable mass based on air consumption
        self.initial_mass = mass  # Store initial mass
        
        # For tracking history
        self.position_history = []
        self.velocity_history = []
        self.thrust_history = []
        self.time_history = []
        self.air_flow_history = []
        self.air_remaining_history = []
        self.simulation_time = 0.0
        
    def apply_control(self, control_input):
        """
        Apply a control input (0 to 1) which will be delayed by the pneumatic system.
        
        Args:
            control_input (float): Control input between 0 (no thrust) and 1 (max thrust)
        """
        # Constrain control input between 0 and 1
        control_input = np.clip(control_input, 0.0, 1.0)
        
        # Check if tank is empty and limit thrust if needed
        if self.enable_air_consumption and self.tank_empty:
            control_input = 0.0
        
        # Add to history for delayed application
        self.control_history.append(control_input)
    
    def step(self):
        """
        Step the simulation forward by dt seconds.
        Returns the current state (position, velocity, acceleration).
        """
        # Get the delayed control input
        delayed_control = self.control_history[0]
        
        # Apply air consumption model to calculate thrust
        if self.enable_air_consumption and self.air_model:
            # Calculate desired thrust from control input
            desired_thrust = delayed_control * self.max_thrust
            
            # Update air consumption model
            self.air_flow_rate, self.air_efficiency, self.air_remaining, self.tank_empty = self.air_model.update(
                delayed_control, desired_thrust
            )
            
            # If tank is empty, no thrust can be applied
            if self.tank_empty:
                self.thrust = 0.0
            else:
                # Calculate actual thrust based on air consumption and efficiency
                self.thrust = desired_thrust * self.air_efficiency
        else:
            # Simple thrust calculation without air consumption model
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
        
        # Record air consumption history if enabled
        if self.enable_air_consumption:
            self.air_flow_history.append(self.air_flow_rate)
            self.air_remaining_history.append(self.air_remaining)
        
        # Return current state
        return self.position, self.velocity, self.acceleration
    
    def get_history(self):
        """Return the history of position, velocity, thrust, and time."""
        history_data = {
            'position': self.position_history,
            'velocity': self.velocity_history,
            'thrust': self.thrust_history,
            'time': self.time_history
        }
        
        # Add air consumption data if available
        if self.enable_air_consumption:
            history_data['air_flow_rate'] = self.air_flow_history
            history_data['air_remaining'] = self.air_remaining_history
            
            # Add summary statistics if model exists
            if self.air_model:
                history_data['air_stats'] = self.air_model.get_summary_statistics()
        
        return history_data
    
    def get_air_consumption_status(self):
        """Get the current air consumption status."""
        if not self.enable_air_consumption or not self.air_model:
            return {
                'enabled': False,
                'flow_rate': 0.0,
                'efficiency': 1.0,
                'remaining': 100.0,
                'empty': False
            }
        
        return {
            'enabled': True,
            'flow_rate': self.air_flow_rate,
            'efficiency': self.air_efficiency,
            'remaining': self.air_remaining,
            'empty': self.tank_empty,
            'pressure': self.air_model.current_pressure,
            'mass': self.air_model.current_air_mass,
            'remaining_time': self.air_model.get_remaining_operation_time()
        }
    
    def plot_air_consumption(self):
        """Generate plots for air consumption if model exists."""
        if self.enable_air_consumption and self.air_model:
            return self.air_model.plot_consumption_metrics()
        return None
    
    def reset(self, initial_position=2.0):
        """Reset the simulation to initial conditions."""
        self.position = initial_position
        self.velocity = 0.0
        self.acceleration = -self.gravity
        self.thrust = 0.0
        self.control_history = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # Reset air consumption model if enabled
        if self.enable_air_consumption and self.air_model:
            self.air_model.reset()
            self.air_flow_rate = 0.0
            self.air_efficiency = 1.0
            self.air_remaining = 100.0
            self.tank_empty = False
        
        # Reset tracking history
        self.position_history = []
        self.velocity_history = []
        self.thrust_history = []
        self.time_history = []
        self.air_flow_history = []
        self.air_remaining_history = []
        self.simulation_time = 0.0
        
        # Reset mass if using variable mass
        if self.variable_mass:
            self.mass = self.initial_mass


# Test code
if __name__ == "__main__":
    # Simple test to verify the physics engine
    engine = PhysicsEngine(enable_air_consumption=True)
    
    # Run for 5 seconds
    total_steps = int(5.0 / engine.dt)
    
    for i in range(total_steps):
        # Apply a simple control strategy for testing
        if i * engine.dt < 2.0:
            engine.apply_control(0.0)  # No thrust for 2 seconds
        else:
            engine.apply_control(0.8)  # 80% thrust after 2 seconds
        
        pos, vel, acc = engine.step()
        
        # Print status every 20 steps
        if i % 20 == 0:
            air_status = engine.get_air_consumption_status()
            print(f"Time: {i*engine.dt:.2f}s, Position: {pos:.2f}m, Velocity: {vel:.2f}m/s")
            if engine.enable_air_consumption:
                print(f"  Air remaining: {air_status['remaining']:.1f}%, Flow rate: {air_status['flow_rate']*1000:.1f} g/s")
    
    # Plot air consumption data
    if engine.enable_air_consumption:
        engine.plot_air_consumption()
        import matplotlib.pyplot as plt
        plt.show()
    
    print("Test complete.")
