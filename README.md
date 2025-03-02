# Pneumatic Hopper Simulation

A physics-based simulation of a self-leveling pneumatic hopper system constrained to vertical movement. This project demonstrates altitude control using delayed hysteresis control and Kalman filtering for state estimation.

## Overview

This simulation models a pneumatically actuated test bed for altitude control, similar to systems that could potentially be used for autonomous vehicles in atmosphere-lacking environments (like the Moon). The hopper uses compressed air thrusters for self-leveling and is constrained to a vertical track.

### Key Features

- Pneumatic thrust system with realistic delay modeling
- Kalman filter implementation for state estimation based on simulated LiDAR and MPU6050 sensor data
- Delayed hysteresis control system for altitude stabilization
- Pygame-based visualization of the system dynamics and control methods

## Physics Model

The system is modeled using a delayed differential equation approach:

```
dx(t)/dt = f(x(t), u(t-τ))
```

Where:
- x(t) represents the system state at time t
- u(t-τ) represents the control input with a delay τ
- The delay models the pneumatic response time due to the tube connecting the compressor to the hopper

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the simulation:

```bash
python main.py
```

## Controls

- **Space**: Toggle thrust
- **Up/Down Arrow Keys**: Adjust target height
- **R**: Reset simulation
- **K**: Toggle Kalman filter visualization
- **H**: Toggle hysteresis control visualization

## Project Structure

- `main.py`: Entry point for the simulation
- `physics.py`: Physical model of the hopper system with pneumatic delay
- `kalman_filter.py`: Implementation of Kalman filter for state estimation
- `controller.py`: Hysteresis controller with built-in delay
- `visualization.py`: Pygame-based visualization module
- `sensor.py`: Simulated sensor data from LiDAR and MPU6050

## Future Improvements

- Advanced air consumption modeling
- Variable mass considerations
- Additional control methods for comparison
- Extended degrees of freedom (2D or 3D movement)
