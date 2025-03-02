import numpy as np

class KalmanFilter:
    """
    Kalman filter implementation for state estimation of the pneumatic hopper.
    
    This uses a constant acceleration model with position and acceleration measurements
    from the LiDAR and MPU6050 sensors.
    """
    
    def __init__(self, dt=0.01, initial_position=2.0, measurement_noise_position=0.05, 
                 measurement_noise_acceleration=0.1, process_noise=0.01):
        """
        Initialize the Kalman filter.
        
        Args:
            dt (float): Time step in seconds
            initial_position (float): Initial position estimate
            measurement_noise_position (float): Standard deviation of position measurement noise
            measurement_noise_acceleration (float): Standard deviation of acceleration measurement noise
            process_noise (float): Process noise parameter
        """
        # State vector: [position, velocity, acceleration]
        self.x = np.array([initial_position, 0.0, -9.81])  # Initial state
        
        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # Process noise covariance matrix
        self.Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1]
        ]) * process_noise
        
        # Measurement matrices
        # For position measurement (LiDAR)
        self.H_position = np.array([[1, 0, 0]])
        
        # For acceleration measurement (MPU6050)
        self.H_acceleration = np.array([[0, 0, 1]])
        
        # Measurement noise covariance matrices
        self.R_position = np.array([[measurement_noise_position**2]])
        self.R_acceleration = np.array([[measurement_noise_acceleration**2]])
        
        # Initial estimate error covariance
        self.P = np.eye(3)
        
        # For tracking state history
        self.state_history = []
        self.position_estimate_history = []
        self.velocity_estimate_history = []
        self.acceleration_estimate_history = []
        self.covariance_history = []
        self.time_history = []
        self.simulation_time = 0.0
        self.dt = dt
    
    def predict(self):
        """
        Perform the prediction step of the Kalman filter.
        """
        # Project the state ahead
        self.x = self.F @ self.x
        
        # Project the error covariance ahead
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update_position(self, position_measurement):
        """
        Update the state estimate with a position measurement.
        
        Args:
            position_measurement (float): Position measurement from LiDAR
        """
        if position_measurement is None:
            return
        
        # Calculate Kalman gain
        K = self.P @ self.H_position.T @ np.linalg.inv(self.H_position @ self.P @ self.H_position.T + self.R_position)
        
        # Update estimate with measurement
        self.x = self.x + K @ (position_measurement - self.H_position @ self.x)
        
        # Update the error covariance
        self.P = (np.eye(3) - K @ self.H_position) @ self.P
    
    def update_acceleration(self, acceleration_measurement):
        """
        Update the state estimate with an acceleration measurement.
        
        Args:
            acceleration_measurement (float): Acceleration measurement from MPU6050
        """
        if acceleration_measurement is None:
            return
        
        # Calculate Kalman gain
        K = self.P @ self.H_acceleration.T @ np.linalg.inv(self.H_acceleration @ self.P @ self.H_acceleration.T + self.R_acceleration)
        
        # Update estimate with measurement
        self.x = self.x + K @ (acceleration_measurement - self.H_acceleration @ self.x)
        
        # Update the error covariance
        self.P = (np.eye(3) - K @ self.H_acceleration) @ self.P
    
    def step(self, position_measurement=None, acceleration_measurement=None):
        """
        Perform a complete filter step (predict and update).
        
        Args:
            position_measurement (float): Position measurement from LiDAR
            acceleration_measurement (float): Acceleration measurement from MPU6050
            
        Returns:
            numpy.ndarray: Updated state estimate [position, velocity, acceleration]
        """
        # Prediction step
        self.predict()
        
        # Update steps with available measurements
        if position_measurement is not None:
            self.update_position(position_measurement)
        
        if acceleration_measurement is not None:
            self.update_acceleration(acceleration_measurement)
        
        # Update simulation time and history
        self.simulation_time += self.dt
        self.state_history.append(self.x.copy())
        self.position_estimate_history.append(self.x[0])
        self.velocity_estimate_history.append(self.x[1])
        self.acceleration_estimate_history.append(self.x[2])
        self.covariance_history.append(np.diag(self.P).copy())  # Store diagonal elements
        self.time_history.append(self.simulation_time)
        
        return self.x
    
    def get_state(self):
        """
        Get the current state estimate.
        
        Returns:
            tuple: (position, velocity, acceleration)
        """
        return self.x[0], self.x[1], self.x[2]
    
    def get_history(self):
        """
        Get the history of state estimates and covariances.
        
        Returns:
            dict: Dictionary with state and covariance history
        """
        return {
            'position': self.position_estimate_history,
            'velocity': self.velocity_estimate_history,
            'acceleration': self.acceleration_estimate_history,
            'covariance': self.covariance_history,
            'time': self.time_history
        }
    
    def reset(self, initial_position=2.0):
        """
        Reset the Kalman filter.
        
        Args:
            initial_position (float): Initial position estimate
        """
        self.x = np.array([initial_position, 0.0, -9.81])
        self.P = np.eye(3)
        self.state_history = []
        self.position_estimate_history = []
        self.velocity_estimate_history = []
        self.acceleration_estimate_history = []
        self.covariance_history = []
        self.time_history = []
        self.simulation_time = 0.0


# Test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple test for the Kalman filter
    dt = 0.01
    simulation_time = 5.0  # seconds
    steps = int(simulation_time / dt)
    
    # Initialize filter
    kf = KalmanFilter(dt=dt)
    
    # True state
    true_position = np.zeros(steps)
    true_velocity = np.zeros(steps)
    true_acceleration = np.zeros(steps)
    
    # Initial conditions
    true_position[0] = 2.0
    true_acceleration[0] = -9.81  # gravity
    
    # Generate true trajectory
    for i in range(1, steps):
        # Simulate thrust after 2 seconds
        if i * dt > 2.0:
            true_acceleration[i] = -9.81 + 15.0  # gravity + thrust
        else:
            true_acceleration[i] = -9.81  # just gravity
        
        true_velocity[i] = true_velocity[i-1] + true_acceleration[i-1] * dt
        true_position[i] = true_position[i-1] + true_velocity[i-1] * dt + 0.5 * true_acceleration[i-1] * dt**2
    
    # Generate noisy measurements
    position_noise_std = 0.05
    acceleration_noise_std = 0.1
    
    # Not every time step gets a measurement
    position_measurement_interval = 5  # every 5 steps
    acceleration_measurement_interval = 2  # every 2 steps
    
    # Estimated states
    estimated_position = np.zeros(steps)
    estimated_velocity = np.zeros(steps)
    estimated_acceleration = np.zeros(steps)
    
    # Run filter
    for i in range(steps):
        # Generate noisy measurements (only at certain intervals)
        position_measurement = None
        if i % position_measurement_interval == 0:
            position_measurement = true_position[i] + np.random.normal(0, position_noise_std)
        
        acceleration_measurement = None
        if i % acceleration_measurement_interval == 0:
            acceleration_measurement = true_acceleration[i] + np.random.normal(0, acceleration_noise_std)
        
        # Step the filter
        kf.step(position_measurement, acceleration_measurement)
        
        # Store estimates
        estimated_position[i], estimated_velocity[i], estimated_acceleration[i] = kf.get_state()
    
    # Plotting
    time = np.arange(0, simulation_time, dt)
    
    plt.figure(figsize=(14, 10))
    
    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(time, true_position, 'b-', label='True Position')
    plt.plot(time, estimated_position, 'r-', label='Estimated Position')
    plt.ylabel('Position (m)')
    plt.grid(True)
    plt.legend()
    
    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(time, true_velocity, 'b-', label='True Velocity')
    plt.plot(time, estimated_velocity, 'r-', label='Estimated Velocity')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    
    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(time, true_acceleration, 'b-', label='True Acceleration')
    plt.plot(time, estimated_acceleration, 'r-', label='Estimated Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kalman_filter_test.png')
    plt.close()
    
    print("Test complete. Results saved to kalman_filter_test.png")
