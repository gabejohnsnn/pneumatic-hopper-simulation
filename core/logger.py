#!/usr/bin/env python3
"""
Logger for the pneumatic hopper simulation.

Handles recording, storing, and saving simulation data for later analysis.
"""

import os
import time
import pickle
import numpy as np
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
        
        # Initialize data storage
        self.reset()
    
    def reset(self):
        """Reset all logged data."""
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
        
        # Remove additional data if it exists
        if hasattr(self, 'additional_data'):
            delattr(self, 'additional_data')
    
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
    
    def log_step(self, log_data):
        """
        Log all data for a single simulation step.
        
        Args:
            log_data (dict): Dictionary containing all data for this step
        """
        # Log physics data
        self.log_physics(
            log_data['time'],
            log_data['true_position'],
            log_data['true_velocity'],
            log_data['true_acceleration'],
            log_data['applied_thrust']
        )
        
        # Log Kalman filter data
        self.log_kalman(
            log_data['time'],
            log_data['est_position'],
            log_data['est_velocity'],
            log_data['est_acceleration'],
            log_data.get('kf_covariance_diag', None)
        )
        
        # Log controller data
        self.log_controller(
            log_data['time'],
            log_data['target_height'],
            log_data['target_height'] - log_data['est_position'],
            log_data['control_output']
        )
        
        # Log sensor data
        self.log_sensors(
            log_data['time'],
            log_data['lidar_reading'],
            log_data['mpu_reading']
        )
        
        # Log additional data if present
        additional_data = {}
        for key, value in log_data.items():
            if key not in ['time', 'true_position', 'true_velocity', 'true_acceleration', 
                          'est_position', 'est_velocity', 'est_acceleration', 
                          'lidar_reading', 'mpu_reading', 'target_height', 
                          'control_output', 'applied_thrust', 'kf_covariance_diag']:
                additional_data[key] = value
        
        if additional_data:
            self.log_additional(log_data['time'], additional_data)
    
    def get_data(self):
        """
        Get all logged data.
        
        Returns:
            dict: Dictionary containing all logged data
        """
        data = {
            'physics': self.physics_data,
            'kalman': self.kalman_data,
            'controller': self.controller_data,
            'sensors': self.sensor_data,
            'timestamp': time.time()
        }
        
        if hasattr(self, 'additional_data'):
            data['additional'] = self.additional_data
        
        return data
    
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
        
        # Get all data
        data = self.get_data()
        
        # Save to file
        filepath = os.path.join(self.log_folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
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
