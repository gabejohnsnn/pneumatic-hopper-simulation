{
  `path`: `README.md`,
  `repo`: `pneumatic-hopper-simulation`,
  `owner`: `gabejohnsnn`,
  `branch`: `main`,
  `content`: `# Pneumatic Hopper Simulation

A physics-based simulation of a self-leveling pneumatic hopper system constrained to vertical movement. This project demonstrates altitude control using various control methods, including delayed hysteresis control, PID control, and bang-bang control with Kalman filtering for state estimation.

## Overview

This simulation models a pneumatically actuated test bed for altitude control, similar to systems that could potentially be used for autonomous vehicles in atmosphere-lacking environments (like the Moon). The hopper uses compressed air thrusters for self-leveling and is constrained to a vertical track.

### Key Features

- Pneumatic thrust system with realistic delay modeling
- Kalman filter implementation for state estimation based on simulated LiDAR and MPU6050 sensor data
- Multiple control approaches:
  - Delayed hysteresis control
  - PID control with anti-windup
  - Bang-Bang control based on Pontryagin's maximum principle
- Interactive parameter adjustment during simulation
- Pygame-based visualization of the system dynamics and control methods
- Comprehensive data logging and analysis tools

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

## Command Line Arguments

The simulation offers various command line arguments for customization:

```
usage: main.py [-h] [--mass MASS] [--thrust THRUST] [--delay DELAY] [--target TARGET] 
               [--control {Hysteresis,PID,Bang-Bang}] [--lidar-noise LIDAR_NOISE] 
               [--mpu-noise MPU_NOISE] [--dt DT] [--no-auto] [--no-params] [--log] 
               [--log-freq LOG_FREQ] [--log-dir LOG_DIR] [--analyze]

options:
  -h, --help            show this help message and exit
  --mass MASS           Hopper mass in kg (default: 1.0)
  --thrust THRUST       Maximum thrust force in N (default: 20.0)
  --delay DELAY         Pneumatic delay time in seconds (default: 0.2)
  --target TARGET       Initial target height in meters (default: 3.0)
  --control {Hysteresis,PID,Bang-Bang}
                        Control method (default: Hysteresis)
  --lidar-noise LIDAR_NOISE
                        LiDAR measurement noise std deviation (default: 0.05)
  --mpu-noise MPU_NOISE
                        MPU6050 measurement noise std deviation (default: 0.1)
  --dt DT               Simulation time step in seconds (default: 0.01)
  --no-auto             Disable automatic control (manual thrust only)
  --no-params           Hide parameter adjustment panel
  --log                 Enable data logging
  --log-freq LOG_FREQ   Logging frequency (1 = every step, 10 = every 10th step)
  --log-dir LOG_DIR     Directory to store log files
  --analyze             Analyze and plot results after simulation ends
```

## Controls

- **Space**: Toggle manual thrust
- **Up/Down Arrow Keys**: Adjust target height
- **R**: Reset simulation
- **K**: Toggle Kalman filter visualization
- **H**: Toggle hysteresis control visualization
- **Esc**: Quit simulation

## Parameter Adjustment

The simulation includes a parameter adjustment panel allowing you to modify various aspects in real-time:

- Physics parameters:
  - Mass
  - Maximum thrust
  - Pneumatic delay
  - Air resistance

- Control parameters:
  - Control method (Hysteresis, PID, Bang-Bang)
  - Method-specific parameters:
    - Hysteresis band width
    - PID gains (P, I, D)
    - Bang-Bang threshold

## Data Analysis

The simulation includes tools for data logging and analysis. To enable logging and analysis:

```bash
python main.py --log --analyze
```

This will:
1. Run the simulation and log all data
2. After the simulation ends, generate plots of:
   - Overall system performance
   - Kalman filter accuracy
   - Controller performance

You can also analyze saved log files separately:

```bash
python analysis.py logs/simulation_YYYYMMDD_HHMMSS.pkl
```

## Project Structure

- `main.py`: Entry point for the simulation
- `physics.py`: Physical model of the hopper system with pneumatic delay
- `sensor.py`: Simulated sensor data from LiDAR and MPU6050
- `kalman_filter.py`: Implementation of Kalman filter for state estimation
- `controllers/`: Control methods package
  - `hysteresis_controller.py`: Hysteresis controller with built-in delay
  - `pid_controller.py`: PID controller with anti-windup
  - `bang_bang_controller.py`: Bang-Bang controller using maximum principle
- `visualization.py`: Pygame-based visualization module
- `parameters.py`: Parameter adjustment GUI module
- `analysis.py`: Data logging and analysis tools

## Future Improvements

- Advanced air consumption modeling
- Variable mass considerations
- Model predictive control (MPC) implementation
- Extended degrees of freedom (2D or 3D movement)
- Hardware-in-the-loop testing capability`
}
