#!/usr/bin/env python3
"""
Analysis and plotting utilities for the pneumatic hopper simulation.

This script can be imported by main.py to log data during simulation,
or run standalone to analyze previously saved data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from datetime import datetime


class SimulationLogger:
    """
    Logger for recording simulation data for later analysis.
    """
    
    def __init__(self, log_folder='logs'):
        """
        Initialize the logger.
        
        Args:
            log_folder (str): Folder to save logs to
        """
        self.log_folder = log_folder
        
        # Create log folder if it doesn't exist
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        # Data storage
        self.physics_data = {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': [],
            'thrust': []
        }
        
        self.kalman_data = {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': [],
            'covariance': []
        }
        
        self.controller_data = {
            'time': [],
            'target': [],
            'error': [],
            'output': []
        }
        
        self.sensor_data = {
            'time': [],
            'lidar_readings': [],
            'mpu_readings': []
        }
    
    def log_physics(self, time, position, velocity, acceleration, thrust):
        """Log physics data."""
        self.physics_data['time'].append(time)
        self.physics_data['position'].append(position)
        self.physics_data['velocity'].append(velocity)
        self.physics_data['acceleration'].append(acceleration)
        self.physics_data['thrust'].append(thrust)

    def log_additional(self, time, data_dict):
        """
        Log additional custom data.
        
        Args:
            time (float): Simulation time
            data_dict (dict): Dictionary with additional data to log
        """
        if not hasattr(self, 'additional_data'):
            self.additional_data = {
                'time': [],
            }
            # Initialize dict entries for each key in data_dict
            for key in data_dict:
                self.additional_data[key] = []
        
        # Store time and data
        self.additional_data['time'].append(time)
        for key, value in data_dict.items():
            if key not in self.additional_data:
                self.additional_data[key] = [None] * (len(self.additional_data['time']) - 1)
            self.additional_data[key].append(value)
    
    def log_kalman(self, time, position, velocity, acceleration, covariance=None):
        """Log Kalman filter estimates."""
        self.kalman_data['time'].append(time)
        self.kalman_data['position'].append(position)
        self.kalman_data['velocity'].append(velocity)
        self.kalman_data['acceleration'].append(acceleration)
        self.kalman_data['covariance'].append(covariance if covariance is not None else [0, 0, 0])
    
    def log_controller(self, time, target, error, output):
        """Log controller data."""
        self.controller_data['time'].append(time)
        self.controller_data['target'].append(target)
        self.controller_data['error'].append(error)
        self.controller_data['output'].append(output)
    
    def log_sensors(self, time, lidar_reading, mpu_reading):
        """Log sensor readings."""
        self.sensor_data['time'].append(time)
        self.sensor_data['lidar_readings'].append(lidar_reading if lidar_reading is not None else np.nan)
        self.sensor_data['mpu_readings'].append(mpu_reading if mpu_reading is not None else np.nan)
    
    def save_data(self, filename=None):
        """
        Save all logged data to a file.
        
        Args:
            filename (str, optional): Filename to save to. If None, a timestamped filename is used.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}.pkl"
        
        # Combine all data
        data = {
            'physics': self.physics_data,
            'kalman': self.kalman_data,
            'controller': self.controller_data,
            'sensors': self.sensor_data,
            'timestamp': time.time()
        }
        
        # Save to file
        filepath = os.path.join(self.log_folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        if hasattr(self, 'additional_data'):
            data['additional'] = self.additional_data
        
        print(f"Simulation data saved to {filepath}")
        return filepath
    
    @staticmethod
    def load_data(filepath):
        """
        Load saved simulation data from a file.
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            dict: Loaded data
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data


def plot_simulation_results(data, title=None, save_path=None):
    """
    Plot simulation results from logged data.
    
    Args:
        data (dict): Simulation data dictionary
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    kalman = data['kalman']
    controller = data['controller']
    sensors = data['sensors']
    
    # Create figure
    fig = plt.figure(figsize=(12, 15))
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
        fig.suptitle(f'Pneumatic Hopper Simulation Results - {timestamp}', fontsize=16)
    
    # Position plot
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(physics['time'], physics['position'], 'b-', label='True Position')
    ax1.plot(kalman['time'], kalman['position'], 'r--', label='Kalman Estimate')
    ax1.plot(controller['time'], controller['target'], 'g-', label='Target')
    
    # Plot sensor readings
    lidar_time = np.array(sensors['time'])
    lidar_readings = np.array(sensors['lidar_readings'])
    valid_lidar = ~np.isnan(lidar_readings)
    if np.any(valid_lidar):
        ax1.scatter(lidar_time[valid_lidar], lidar_readings[valid_lidar], 
                   marker='.', color='g', s=10, alpha=0.5, label='LiDAR')
    
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Velocity plot
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(physics['time'], physics['velocity'], 'b-', label='True Velocity')
    ax2.plot(kalman['time'], kalman['velocity'], 'r--', label='Kalman Estimate')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    # Acceleration plot
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(physics['time'], physics['acceleration'], 'b-', label='True Acceleration')
    ax3.plot(kalman['time'], kalman['acceleration'], 'r--', label='Kalman Estimate')
    
    # Plot sensor readings
    mpu_time = np.array(sensors['time'])
    mpu_readings = np.array(sensors['mpu_readings'])
    valid_mpu = ~np.isnan(mpu_readings)
    if np.any(valid_mpu):
        ax3.scatter(mpu_time[valid_mpu], mpu_readings[valid_mpu], 
                   marker='.', color='m', s=10, alpha=0.5, label='MPU6050')
    
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.grid(True)
    ax3.legend()
    
    # Control output plot
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(controller['time'], controller['output'], 'k-', label='Control Output')
    ax4.plot(physics['time'], physics['thrust'], 'b-', label='Applied Thrust')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control / Thrust')
    ax4.grid(True)
    ax4.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_kalman_performance(data, save_path=None):
    """
    Plot Kalman filter performance analysis.
    
    Args:
        data (dict): Simulation data dictionary
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    kalman = data['kalman']
    
    # Interpolate physics data to match Kalman timestamps
    # This is necessary because physics and Kalman data might be logged at different rates
    from scipy.interpolate import interp1d
    
    # Make sure we have data to interpolate
    if len(physics['time']) < 2 or len(kalman['time']) < 2:
        print("Not enough data points for Kalman performance analysis")
        return
    
    # Create interpolation functions
    interp_pos = interp1d(physics['time'], physics['position'], bounds_error=False, fill_value='extrapolate')
    interp_vel = interp1d(physics['time'], physics['velocity'], bounds_error=False, fill_value='extrapolate')
    interp_acc = interp1d(physics['time'], physics['acceleration'], bounds_error=False, fill_value='extrapolate')
    
    # Interpolate physics data at Kalman timestamps
    true_pos = interp_pos(kalman['time'])
    true_vel = interp_vel(kalman['time'])
    true_acc = interp_acc(kalman['time'])
    
    # Calculate errors
    pos_error = np.array(kalman['position']) - true_pos
    vel_error = np.array(kalman['velocity']) - true_vel
    acc_error = np.array(kalman['acceleration']) - true_acc
    
    # Extract covariance diagonal (variances)
    covariance = np.array(kalman['covariance'])
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Kalman Filter Performance Analysis', fontsize=16)
    
    # Position error plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(kalman['time'], pos_error, 'b-', label='Position Error')
    if covariance.shape[1] >= 1:
        std_pos = np.sqrt(covariance[:, 0])
        ax1.fill_between(kalman['time'], -2*std_pos, 2*std_pos, color='b', alpha=0.2, label='±2σ')
    ax1.set_ylabel('Position Error (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Velocity error plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(kalman['time'], vel_error, 'g-', label='Velocity Error')
    if covariance.shape[1] >= 2:
        std_vel = np.sqrt(covariance[:, 1])
        ax2.fill_between(kalman['time'], -2*std_vel, 2*std_vel, color='g', alpha=0.2, label='±2σ')
    ax2.set_ylabel('Velocity Error (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    # Acceleration error plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(kalman['time'], acc_error, 'r-', label='Acceleration Error')
    if covariance.shape[1] >= 3:
        std_acc = np.sqrt(covariance[:, 2])
        ax3.fill_between(kalman['time'], -2*std_acc, 2*std_acc, color='r', alpha=0.2, label='±2σ')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration Error (m/s²)')
    ax3.grid(True)
    ax3.legend()
    
    # Print error statistics
    rms_pos = np.sqrt(np.mean(pos_error**2))
    rms_vel = np.sqrt(np.mean(vel_error**2))
    rms_acc = np.sqrt(np.mean(acc_error**2))
    
    print(f"Kalman Filter Performance:")
    print(f"  Position RMS Error: {rms_pos:.4f} m")
    print(f"  Velocity RMS Error: {rms_vel:.4f} m/s")
    print(f"  Acceleration RMS Error: {rms_acc:.4f} m/s²")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Kalman performance plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_controller_performance(data, save_path=None):
    """
    Plot controller performance analysis.
    
    Args:
        data (dict): Simulation data dictionary
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    controller = data['controller']
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Controller Performance Analysis', fontsize=16)
    
    # Position and target plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(physics['time'], physics['position'], 'b-', label='Position')
    ax1.plot(controller['time'], controller['target'], 'r--', label='Target')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Error plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(controller['time'], controller['error'], 'g-', label='Error')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Error (m)')
    ax2.grid(True)
    ax2.legend()
    
    # Control output and thrust plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(controller['time'], controller['output'], 'k-', label='Control Output')
    ax3.plot(physics['time'], physics['thrust'], 'b-', label='Applied Thrust')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control / Thrust')
    ax3.grid(True)
    ax3.legend()
    
    # Print performance statistics
    settling_time = None
    overshoot = None
    steady_state_error = None
    
    # Calculate settling time (time to reach within 5% of target)
    if len(controller['time']) > 0 and len(controller['error']) > 0:
        # Get final target
        final_target = controller['target'][-1]
        
        # Calculate 5% threshold
        threshold = 0.05 * final_target
        
        # Find settling time
        for i in range(len(controller['time'])):
            if abs(controller['error'][i]) <= threshold:
                # Check if it stays within threshold
                if all(abs(e) <= threshold for e in controller['error'][i:]):
                    settling_time = controller['time'][i] - controller['time'][0]
                    break
        
        # Calculate overshoot
        max_pos = max(physics['position'])
        if max_pos > final_target:
            overshoot = (max_pos - final_target) / final_target * 100
        else:
            overshoot = 0.0
        
        # Calculate steady-state error (average of last 10% of data)
        n = len(controller['error'])
        if n > 10:
            steady_state_error = np.mean(np.abs(controller['error'][int(0.9*n):]))
    
    print(f"Controller Performance:")
    if settling_time is not None:
        print(f"  Settling Time (5%): {settling_time:.2f} s")
    if overshoot is not None:
        print(f"  Overshoot: {overshoot:.2f}%")
    if steady_state_error is not None:
        print(f"  Steady-State Error: {steady_state_error:.4f} m")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Controller performance plot saved to {save_path}")
    
    # Show plot
    plt.show()


# If run as a standalone script
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pneumatic Hopper Simulation Analysis')
    parser.add_argument('file', help='Simulation data file to analyze')
    parser.add_argument('--output', '-o', help='Output folder for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    args = parser.parse_args()
    
    # Load data
    data = SimulationLogger.load_data(args.file)
    
    # Create output folder if needed
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Generate plots
    if args.output:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        save_path = os.path.join(args.output, f"{base_name}_results.png")
        kalman_path = os.path.join(args.output, f"{base_name}_kalman.png")
        controller_path = os.path.join(args.output, f"{base_name}_controller.png")
    else:
        save_path = None
        kalman_path = None
        controller_path = None
    
    # Plot results
    if not args.no_show or save_path:
        plot_simulation_results(data, save_path=save_path)
    
    # Plot Kalman performance
    if not args.no_show or kalman_path:
        plot_kalman_performance(data, save_path=kalman_path)
    
    # Plot controller performance
    if not args.no_show or controller_path:
        plot_controller_performance(data, save_path=controller_path)
