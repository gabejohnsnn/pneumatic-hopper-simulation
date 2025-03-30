A physics-based simulation of a self-leveling pneumatic hopper system constrained to vertical movement. This project demonstrates altitude control using various control methods, including delayed hysteresis control, PID control, bang-bang control, and reinforcement learning with DDPG (Deep Deterministic Policy Gradient) combined with Kalman filtering for state estimation.

## Overview

This simulation models a pneumatically actuated test bed for altitude control, similar to systems that could potentially be used for autonomous vehicles in atmosphere-lacking environments (like the Moon). The hopper uses compressed air thrusters for self-leveling and is constrained to a vertical track.

### Key Features

- Pneumatic thrust system with realistic delay modeling
- Kalman filter implementation for state estimation based on simulated LiDAR and MPU6050 sensor data
- Multiple control approaches:
  - Delayed hysteresis control
  - PID control with anti-windup
  - Bang-Bang control based on Pontryagin's maximum principle
  - DDPG reinforcement learning control (model-free learning)
  - Model Predictive Control (MPC) for advanced prediction-based control
- Interactive parameter adjustment during simulation
- Pygame-based visualization of the system dynamics and control methods
- Comprehensive data logging and analysis tools
- **NEW**: Controller comparison functionality for quantitative evaluation

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
               [--controllers {Hysteresis,PID,BangBang,DDPG,MPC} [{Hysteresis,PID,BangBang,DDPG,MPC} ...]]
               [--lidar-noise LIDAR_NOISE] [--mpu-noise MPU_NOISE] [--dt DT] [--duration DURATION]
               [--no-auto] [--headless] [--log] [--log-freq LOG_FREQ] [--log-dir LOG_DIR] 
               [--analyze] [--compare] [--ddpg-load DDPG_LOAD] [--ddpg-save DDPG_SAVE] [--ddpg-no-train]

options:
  -h, --help            show this help message and exit
  --mass MASS           Hopper mass in kg (default: 1.0)
  --thrust THRUST       Maximum thrust force in N (default: 20.0)
  --delay DELAY         Pneumatic delay time in seconds (default: 0.2)
  --target TARGET       Initial target height in meters (default: 3.0)
  --controllers {Hysteresis,PID,BangBang,DDPG,MPC} [{Hysteresis,PID,BangBang,DDPG,MPC} ...]
                        Control methods to simulate (default: Hysteresis)
  --lidar-noise LIDAR_NOISE
                        LiDAR measurement noise std deviation (default: 0.05)
  --mpu-noise MPU_NOISE
                        MPU6050 measurement noise std deviation (default: 0.1)
  --dt DT               Simulation time step in seconds (default: 0.01)
  --duration DURATION   Simulation duration in seconds (default: 20.0)
  --no-auto             Disable automatic control (manual thrust only)
  --headless            Run in headless mode (no visualization)
  --log                 Enable data logging
  --log-freq LOG_FREQ   Logging frequency (1 = every step, 10 = every 10th step)
  --log-dir LOG_DIR     Directory to store log files
  --analyze             Analyze and plot results after simulation ends
  --compare             Compare multiple controllers (requires --controllers with multiple values)
  --ddpg-load DDPG_LOAD Load pre-trained DDPG model from file
  --ddpg-save DDPG_SAVE Save trained DDPG model to file when simulation ends
  --ddpg-no-train       Disable DDPG training (evaluation mode only)
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
  - Control method (Hysteresis, PID, Bang-Bang, DDPG, MPC)
  - Method-specific parameters:
    - Hysteresis band width
    - PID gains (P, I, D)
    - Bang-Bang threshold
    - DDPG learning rate and exploration noise
    - MPC prediction horizon

## Controller Comparison

The new refactored structure makes it easy to run multiple controllers and compare their performance:

```bash
# Run and compare multiple controllers
python main.py --controllers Hysteresis PID MPC --compare --analyze --log --duration 30.0
```

This will:
1. Run simulations for each specified controller
2. Log the results for each controller
3. Generate comparison plots showing:
   - Position tracking performance
   - Control effort
   - Settling time
   - Steady-state error

## DDPG Reinforcement Learning

The DDPG controller implements a state-of-the-art model-free reinforcement learning approach for altitude control:

- **Actor-Critic Architecture**: Combines policy gradients with value function approximation for stable learning
- **Experience Replay**: Stores and samples past experiences to break correlations and improve learning stability
- **Exploration via Noise**: Uses Ornstein-Uhlenbeck process to add time-correlated exploration noise
- **Target Networks**: Employs separate target networks with soft updates to stabilize training

To use the DDPG controller:

```bash
# Run with DDPG controller and save the trained model
python main.py --controllers DDPG --log --analyze --ddpg-save models/trained_ddpg.pth

# Load a previously trained model for evaluation (no training)
python main.py --controllers DDPG --ddpg-load models/trained_ddpg.pth --ddpg-no-train
```

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
   - DDPG learning curve (when using DDPG)

You can also analyze saved log files separately:

```bash
python analysis.py logs/simulation_YYYYMMDD_HHMMSS.pkl
```

To compare multiple saved log files:

```bash
python analysis.py logs/ --compare
```

## Project Structure

- `main.py`: Entry point for the simulation
- `physics.py`: Physical model of the hopper system with pneumatic delay
- `sensor.py`: Simulated sensor data from LiDAR and MPU6050
- `kalman_filter.py`: Implementation of Kalman filter for state estimation
- `core/`: Core simulation components
  - `simulation_runner.py`: Central simulation loop and controller management
  - `logger.py`: Logging and data management
- `controllers/`: Control methods package
  - `hysteresis_controller.py`: Hysteresis controller with built-in delay
  - `pid_controller.py`: PID controller with anti-windup
  - `bang_bang_controller.py`: Bang-Bang controller using maximum principle
  - `ddpg_controller.py`: Deep Deterministic Policy Gradient reinforcement learning controller
  - `mpc_controller.py`: Model Predictive Controller with receding horizon
- `visualization.py`: Pygame-based visualization module
- `parameters.py`: Parameter adjustment GUI module
- `analysis.py`: Analysis and plotting tools (single and comparison)

## Requirements

The simulation requires the following dependencies:

```
numpy
pygame
pygame_gui
matplotlib
scipy
torch
```

## Future Improvements

- Advanced air consumption modeling
- Variable mass considerations
- Additional reinforcement learning approaches (PPO, SAC)
- Extended degrees of freedom (2D or 3D movement)
- Hardware-in-the-loop testing capability
