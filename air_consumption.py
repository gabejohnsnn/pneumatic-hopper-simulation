#!/usr/bin/env python3
"""
Air Consumption Model for Pneumatic Hopper Simulation

This module implements a sophisticated model for pneumatic air consumption
based on physical principles of compressed air systems. It accounts for:
- Variable mass flow rate based on thrust level
- Pressure-dependent efficiency
- Realistic tank depletion effects
- Temperature effects on air density
- Choked flow through nozzles
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class AirConsumptionModel:
    """
    Models air consumption for pneumatic thrusters with realistic physics.
    
    This class tracks the consumption of compressed air, models tank pressure
    changes during operation, and provides functions to estimate remaining
    operation time and efficiency metrics.
    """
    
    def __init__(self, 
                 tank_volume=0.02,         # m³ (20L tank)
                 initial_pressure=20e6,    # Pa (200 bar)
                 nozzle_diameter=0.005,    # m (5mm diameter)
                 ambient_pressure=101325,  # Pa (1 atm)
                 initial_temperature=293,  # K (20°C)
                 max_thrust=20.0,          # N
                 specific_gas_constant=287.058,  # J/(kg·K) for air
                 discharge_coefficient=0.85,     # Nozzle efficiency factor
                 specific_heat_ratio=1.4,        # For air (cp/cv)
                 time_window=5.0,          # Time window for flow rate averaging
                 dt=0.01):                 # Simulation time step
        """
        Initialize the air consumption model.
        
        Args:
            tank_volume: Volume of compressed air tank in m³
            initial_pressure: Initial tank pressure in Pa
            nozzle_diameter: Diameter of thruster nozzle in m
            ambient_pressure: Ambient pressure in Pa
            initial_temperature: Initial gas temperature in K
            max_thrust: Maximum thrust capability in N
            specific_gas_constant: Specific gas constant in J/(kg·K)
            discharge_coefficient: Nozzle efficiency factor (0-1)
            specific_heat_ratio: Ratio of specific heats (cp/cv)
            time_window: Time window for flow rate averaging in seconds
            dt: Simulation time step in seconds
        """
        # Tank parameters
        self.tank_volume = tank_volume
        self.current_pressure = initial_pressure
        self.initial_pressure = initial_pressure
        self.temperature = initial_temperature
        self.ambient_pressure = ambient_pressure
        
        # Nozzle parameters
        self.nozzle_diameter = nozzle_diameter
        self.nozzle_area = np.pi * (nozzle_diameter/2)**2
        self.discharge_coefficient = discharge_coefficient
        
        # Gas properties
        self.specific_gas_constant = specific_gas_constant
        self.specific_heat_ratio = specific_heat_ratio
        self.max_thrust = max_thrust
        
        # Critical pressure ratio for choked flow
        self.critical_pressure_ratio = (2 / (self.specific_heat_ratio + 1))**(self.specific_heat_ratio / (self.specific_heat_ratio - 1))
        
        # State tracking
        self.total_air_used = 0.0  # kg
        self.initial_air_mass = self._calculate_air_mass(initial_pressure, initial_temperature)
        self.current_air_mass = self.initial_air_mass
        self.is_tank_empty = False
        self.dt = dt
        
        # Time window for flow rate averaging
        self.time_window_steps = int(time_window / dt)
        self.flow_rate_history = deque(maxlen=self.time_window_steps)
        self.thrust_history = deque(maxlen=self.time_window_steps)
        self.time_history = []
        self.pressure_history = []
        self.flow_rate_history_list = []
        self.efficiency_history = []
        self.simulation_time = 0.0
        
        # For efficiency tracking
        self.total_thrust_produced = 0.0
        self.ideal_specific_impulse = self._calculate_ideal_specific_impulse()
        
        print(f"Air Consumption Model Initialized")
        print(f"Initial air mass: {self.initial_air_mass:.2f} kg")
        print(f"Ideal specific impulse: {self.ideal_specific_impulse:.2f} s")
        print(f"Critical pressure ratio: {self.critical_pressure_ratio:.4f}")
    
    def _calculate_air_mass(self, pressure, temperature):
        """
        Calculate air mass using ideal gas law: m = PV/RT
        
        Args:
            pressure: Air pressure in Pa
            temperature: Air temperature in K
            
        Returns:
            float: Mass of air in kg
        """
        return (pressure * self.tank_volume) / (self.specific_gas_constant * temperature)
    
    def _calculate_ideal_specific_impulse(self):
        """
        Calculate the theoretical specific impulse for the nozzle configuration.
        
        Returns:
            float: Ideal specific impulse in seconds
        """
        # Using the rocket equation for ideal specific impulse
        exit_velocity = np.sqrt(
            (2 * self.specific_heat_ratio * self.specific_gas_constant * self.temperature) /
            (self.specific_heat_ratio - 1) * 
            (1 - (self.ambient_pressure / self.initial_pressure)**((self.specific_heat_ratio - 1) / self.specific_heat_ratio))
        )
        return exit_velocity / 9.81  # Convert to seconds of specific impulse
    
    def _calculate_mass_flow_rate(self, control_input):
        """
        Calculate mass flow rate based on control input and current pressure.
        Models choked/unchoked flow through the nozzle.
        
        Args:
            control_input: Control input between 0 and 1
            
        Returns:
            float: Mass flow rate in kg/s
        """
        if control_input <= 0 or self.is_tank_empty:
            return 0.0
        
        # Adjust nozzle area based on control input (proportional valve model)
        effective_area = self.nozzle_area * control_input * self.discharge_coefficient
        
        # Check if flow is choked (sonic) or unchoked (subsonic)
        pressure_ratio = self.ambient_pressure / self.current_pressure
        
        if pressure_ratio <= self.critical_pressure_ratio:
            # Choked flow (sonic flow, throat velocity = local speed of sound)
            mass_flow = effective_area * self.current_pressure * np.sqrt(
                self.specific_heat_ratio / (self.specific_gas_constant * self.temperature) *
                (2 / (self.specific_heat_ratio + 1))**((self.specific_heat_ratio + 1) / (self.specific_heat_ratio - 1))
            )
        else:
            # Unchoked flow (subsonic)
            mass_flow = effective_area * self.current_pressure * np.sqrt(
                (2 * self.specific_heat_ratio) / (self.specific_gas_constant * self.temperature * (self.specific_heat_ratio - 1)) *
                (pressure_ratio**(2/self.specific_heat_ratio) - pressure_ratio**((self.specific_heat_ratio+1)/self.specific_heat_ratio))
            )
        
        return mass_flow
    
    def _update_pressure(self, mass_consumed):
        """
        Update tank pressure based on air mass consumed.
        
        Args:
            mass_consumed: Mass of air consumed in kg
        """
        if self.is_tank_empty:
            self.current_pressure = self.ambient_pressure
            return
        
        # Update remaining air mass
        self.current_air_mass -= mass_consumed
        
        # Prevent negative mass (shouldn't happen in normal operation)
        if self.current_air_mass <= 0:
            self.current_air_mass = 0
            self.is_tank_empty = True
            self.current_pressure = self.ambient_pressure
            print("Warning: Air tank is empty!")
            return
        
        # Calculate new pressure using ideal gas law (assumes isothermal process for simplicity)
        self.current_pressure = (self.current_air_mass * self.specific_gas_constant * self.temperature) / self.tank_volume
    
    def calculate_thrust_from_flow(self, mass_flow):
        """
        Calculate thrust produced from a given mass flow rate.
        
        Args:
            mass_flow: Mass flow rate in kg/s
            
        Returns:
            float: Thrust force in N
        """
        # Using the thrust equation: F = m_dot * v_exit + (p_exit - p_ambient) * A_exit
        # For simplicity, we're focusing on the momentum term and ignoring pressure term
        
        # Estimate exit velocity based on current pressure ratio
        pressure_ratio = self.ambient_pressure / self.current_pressure
        
        if pressure_ratio <= self.critical_pressure_ratio:
            # Choked flow - exit velocity is sonic
            exit_velocity = np.sqrt(self.specific_heat_ratio * self.specific_gas_constant * self.temperature)
        else:
            # Unchoked flow - exit velocity is subsonic
            exit_velocity = np.sqrt(
                (2 * self.specific_heat_ratio * self.specific_gas_constant * self.temperature) /
                (self.specific_heat_ratio - 1) * 
                (1 - pressure_ratio**((self.specific_heat_ratio - 1) / self.specific_heat_ratio))
            )
        
        # Calculate thrust
        thrust = mass_flow * exit_velocity
        
        return thrust
    
    def calculate_flow_from_thrust(self, desired_thrust, control_input):
        """
        Calculate the mass flow rate needed to produce a desired thrust.
        
        Args:
            desired_thrust: Desired thrust in N
            control_input: Control input between 0 and 1
            
        Returns:
            float: Required mass flow rate in kg/s
        """
        if desired_thrust <= 0 or control_input <= 0:
            return 0.0
        
        # Calculate mass flow directly
        mass_flow = self._calculate_mass_flow_rate(control_input)
        
        # Calculate actual thrust from this flow
        actual_thrust = self.calculate_thrust_from_flow(mass_flow)
        
        # Scale mass flow to match desired thrust
        if actual_thrust > 0:
            mass_flow *= (desired_thrust / actual_thrust)
        
        return mass_flow
    
    def get_flow_efficiency(self):
        """
        Calculate the current efficiency of air consumption.
        
        Returns:
            float: Efficiency as a ratio of actual to ideal specific impulse (0-1)
        """
        # Calculate average thrust and flow rate over time window
        if not self.thrust_history or not self.flow_rate_history:
            return 1.0  # Default to ideal efficiency if no history
        
        avg_thrust = np.mean(list(self.thrust_history))
        avg_flow_rate = np.mean(list(self.flow_rate_history))
        
        if avg_flow_rate <= 1e-6:  # Avoid division by zero
            return 1.0
        
        # Calculate actual specific impulse
        actual_specific_impulse = avg_thrust / (avg_flow_rate * 9.81)
        
        # Calculate efficiency ratio
        efficiency = actual_specific_impulse / self.ideal_specific_impulse
        
        # Bound between 0 and 1
        return np.clip(efficiency, 0.0, 1.0)
    
    def get_remaining_operation_time(self, average_control_input=0.5):
        """
        Estimate the remaining operation time based on current usage patterns.
        
        Args:
            average_control_input: Assumed future average control input (0-1)
            
        Returns:
            float: Estimated remaining operation time in seconds
        """
        if self.is_tank_empty:
            return 0.0
        
        # Calculate average flow rate from recent history
        if self.flow_rate_history:
            avg_flow_rate = np.mean(list(self.flow_rate_history))
        else:
            # Estimate based on average control input
            avg_flow_rate = self._calculate_mass_flow_rate(average_control_input)
        
        if avg_flow_rate <= 1e-6:  # Avoid division by zero
            return float('inf')  # Effectively infinite time left at zero consumption
        
        # Calculate remaining time
        remaining_time = self.current_air_mass / avg_flow_rate
        
        return remaining_time
    
    def update(self, control_input, thrust_produced):
        """
        Update the air consumption model based on the current control input.
        
        Args:
            control_input: Control input between 0 and 1
            thrust_produced: Actual thrust produced in N
        
        Returns:
            tuple: (mass_flow_rate, efficiency, remaining_percentage, is_empty)
        """
        # Skip update if tank is already empty
        if self.is_tank_empty:
            return 0.0, 0.0, 0.0, True
        
        # Calculate current mass flow rate based on control input
        mass_flow_rate = self._calculate_mass_flow_rate(control_input)
        
        # Calculate mass consumed in this time step
        mass_consumed = mass_flow_rate * self.dt
        
        # Update tracking metrics
        self.total_air_used += mass_consumed
        self.total_thrust_produced += thrust_produced * self.dt
        
        # Update tank pressure
        self._update_pressure(mass_consumed)
        
        # Update history for statistics
        self.flow_rate_history.append(mass_flow_rate)
        self.thrust_history.append(thrust_produced)
        
        # Calculate current efficiency
        efficiency = self.get_flow_efficiency()
        
        # Calculate percentage of air remaining
        remaining_percentage = (self.current_air_mass / self.initial_air_mass) * 100.0
        
        # Update simulation time and histories for plotting
        self.simulation_time += self.dt
        self.time_history.append(self.simulation_time)
        self.pressure_history.append(self.current_pressure)
        self.flow_rate_history_list.append(mass_flow_rate)
        self.efficiency_history.append(efficiency)
        
        return mass_flow_rate, efficiency, remaining_percentage, self.is_tank_empty
    
    def plot_consumption_metrics(self):
        """
        Plot air consumption metrics over time.
        
        Returns:
            matplotlib.figure.Figure: Figure object with all plots
        """
        if not self.time_history:
            print("No data to plot yet.")
            return None
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot tank pressure
        axs[0].plot(self.time_history, [p/1e6 for p in self.pressure_history], 'b-')
        axs[0].set_ylabel('Pressure (MPa)')
        axs[0].set_title('Tank Pressure Over Time')
        axs[0].grid(True)
        
        # Plot flow rate
        axs[1].plot(self.time_history, self.flow_rate_history_list, 'r-')
        axs[1].set_ylabel('Flow Rate (kg/s)')
        axs[1].set_title('Mass Flow Rate Over Time')
        axs[1].grid(True)
        
        # Plot efficiency
        axs[2].plot(self.time_history, self.efficiency_history, 'g-')
        axs[2].set_ylabel('Efficiency')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('Flow Efficiency Over Time')
        axs[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def get_summary_statistics(self):
        """
        Get summary statistics of the air consumption.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        stats = {
            'initial_air_mass': self.initial_air_mass,
            'current_air_mass': self.current_air_mass,
            'total_air_used': self.total_air_used,
            'air_used_percentage': (self.total_air_used / self.initial_air_mass) * 100.0 if self.initial_air_mass > 0 else 0,
            'initial_pressure_mpa': self.initial_pressure / 1e6,
            'current_pressure_mpa': self.current_pressure / 1e6,
            'pressure_drop_percentage': ((self.initial_pressure - self.current_pressure) / self.initial_pressure) * 100.0,
            'average_flow_rate': np.mean(self.flow_rate_history_list) if self.flow_rate_history_list else 0,
            'max_flow_rate': max(self.flow_rate_history_list) if self.flow_rate_history_list else 0,
            'average_efficiency': np.mean(self.efficiency_history) if self.efficiency_history else 1.0,
            'is_tank_empty': self.is_tank_empty,
            'estimated_remaining_time': self.get_remaining_operation_time()
        }
        
        # Calculate specific impulse if we've used any air
        if self.total_air_used > 0:
            stats['actual_specific_impulse'] = self.total_thrust_produced / (self.total_air_used * 9.81)
            stats['isp_efficiency'] = stats['actual_specific_impulse'] / self.ideal_specific_impulse
        else:
            stats['actual_specific_impulse'] = 0
            stats['isp_efficiency'] = 0
        
        return stats
    
    def reset(self):
        """Reset the model to initial conditions."""
        self.current_pressure = self.initial_pressure
        self.current_air_mass = self.initial_air_mass
        self.total_air_used = 0.0
        self.total_thrust_produced = 0.0
        self.is_tank_empty = False
        self.flow_rate_history.clear()
        self.thrust_history.clear()
        self.time_history = []
        self.pressure_history = []
        self.flow_rate_history_list = []
        self.efficiency_history = []
        self.simulation_time = 0.0


# Test code
if __name__ == "__main__":
    # Simple test to verify the air consumption model
    model = AirConsumptionModel()
    
    # Parameters for test
    test_duration = 30.0  # seconds
    dt = 0.01
    steps = int(test_duration / dt)
    
    # Run test simulation
    print("\nRunning test simulation...")
    
    for i in range(steps):
        # Simple sinusoidal control input pattern
        time = i * dt
        control = 0.5 + 0.5 * np.sin(time * 0.5)
        
        # Calculate thrust (simplified for test)
        thrust = control * 20.0  # Max thrust is 20N
        
        # Update model
        flow, efficiency, remaining, empty = model.update(control, thrust)
        
        # Print status every second
        if i % 100 == 0:
            print(f"Time: {time:.1f}s | Control: {control:.2f} | Flow: {flow*1000:.1f} g/s | " +
                  f"Efficiency: {efficiency:.2f} | Remaining: {remaining:.1f}%")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    stats = model.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot metrics
    fig = model.plot_consumption_metrics()
    plt.show()
