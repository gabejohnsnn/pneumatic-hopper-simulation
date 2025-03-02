import numpy as np

class SensorSystem:
    """
    Simulates sensors for the pneumatic hopper system.
    This includes:
    - LiDAR for position measurement
    - MPU6050 for acceleration measurement
    """
    
    def __init__(self, lidar_noise_std=0.05, mpu_noise_std=0.1, lidar_update_rate=10, mpu_update_rate=100):
        """
        Initialize the sensor system.
        
        Args:
            lidar_noise_std (float): Standard deviation of LiDAR measurement noise in meters
            mpu_noise_std (float): Standard deviation of MPU6050 acceleration measurement noise in m/s²
            lidar_update_rate (float): LiDAR update rate in Hz
            mpu_update_rate (float): MPU6050 update rate in Hz
        """
        self.lidar_noise_std = lidar_noise_std
        self.mpu_noise_std = mpu_noise_std
        
        # Calculate update intervals in seconds
        self.lidar_update_interval = 1.0 / lidar_update_rate
        self.mpu_update_interval = 1.0 / mpu_update_rate
        
        # Timestamps for last updates
        self.last_lidar_update = 0.0
        self.last_mpu_update = 0.0
        
        # Latest sensor readings
        self.lidar_reading = None
        self.mpu_reading = None
        
        # Flag for new readings
        self.new_lidar_reading = False
        self.new_mpu_reading = False
    
    def update(self, time, position, acceleration):
        """
        Update sensor readings based on true state and current time.
        
        Args:
            time (float): Current simulation time in seconds
            position (float): True position (height) in meters
            acceleration (float): True acceleration in m/s²
            
        Returns:
            tuple: (position_measurement, acceleration_measurement, new_lidar, new_mpu)
                  Each measurement is None if no new reading was taken.
        """
        # Reset new reading flags
        self.new_lidar_reading = False
        self.new_mpu_reading = False
        
        # Check if it's time for a LiDAR update
        if time - self.last_lidar_update >= self.lidar_update_interval:
            # Add Gaussian noise to true position
            self.lidar_reading = position + np.random.normal(0, self.lidar_noise_std)
            self.last_lidar_update = time
            self.new_lidar_reading = True
        
        # Check if it's time for an MPU6050 update
        if time - self.last_mpu_update >= self.mpu_update_interval:
            # Add Gaussian noise to true acceleration
            self.mpu_reading = acceleration + np.random.normal(0, self.mpu_noise_std)
            self.last_mpu_update = time
            self.new_mpu_reading = True
        
        return (
            self.lidar_reading if self.new_lidar_reading else None,
            self.mpu_reading if self.new_mpu_reading else None,
            self.new_lidar_reading,
            self.new_mpu_reading
        )
    
    def get_latest_readings(self):
        """
        Get the latest sensor readings.
        
        Returns:
            tuple: (latest_lidar_reading, latest_mpu_reading)
        """
        return self.lidar_reading, self.mpu_reading
    
    def reset(self):
        """Reset the sensor system."""
        self.last_lidar_update = 0.0
        self.last_mpu_update = 0.0
        self.lidar_reading = None
        self.mpu_reading = None
        self.new_lidar_reading = False
        self.new_mpu_reading = False


# Test code
if __name__ == "__main__":
    # Simple test to verify the sensor system
    sensors = SensorSystem()
    
    # Simulate sensor readings for different positions and accelerations
    test_times = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    test_positions = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    test_accelerations = [9.81, 9.85, 9.9, 9.95, 10.0, 10.05]
    
    for t, p, a in zip(test_times, test_positions, test_accelerations):
        pos_reading, acc_reading, new_lidar, new_mpu = sensors.update(t, p, a)
        
        print(f"Time: {t:.2f}s")
        if new_lidar:
            print(f"  New LiDAR reading: {pos_reading:.4f}m (True: {p:.4f}m)")
        if new_mpu:
            print(f"  New MPU reading: {acc_reading:.4f}m/s² (True: {a:.4f}m/s²)")
    
    print("Test complete.")
