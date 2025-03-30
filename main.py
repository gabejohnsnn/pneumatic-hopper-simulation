#!/usr/bin/env python3
"""
Pneumatic Hopper Simulation - Main Module

This program simulates a pneumatically actuated hopper constrained to vertical movement.
It implements a physics model with pneumatic delay, Kalman filtering for state estimation,
and configurable control methods for altitude control.

Run this script to start the simulation.
"""

import time
import numpy as np
import pygame
import argparse
import os

from physics import PhysicsEngine
from sensor import SensorSystem
from kalman_filter import KalmanFilter
from visualization import Visualizer
from core.logger import SimulationLogger
from core.simulation_runner import SimulationRunner
from analysis import (
    plot_simulation_results, 
    plot_kalman_performance, 
    plot_controller_performance,
    plot_simulation_results_comparison,
    plot_controller_performance_comparison
)

# Import the controller factory function
from controllers import create_controller
from controllers.ddpg_controller import DDPGController


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pneumatic Hopper Simulation')
    
    # Physics parameters
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Hopper mass in kg (default: 1.0)')
    parser.add_argument('--thrust', type=float, default=20.0,
                        help='Maximum thrust force in N (default: 20.0)')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Pneumatic delay time in seconds (default: 0.2)')
    
    # Controller parameters
    parser.add_argument('--target', type=float, default=3.0,
                        help='Initial target height in meters (default: 3.0)')
    parser.add_argument('--controllers', type=str, nargs='+', default=['Hysteresis'],
                        choices=['Hysteresis', 'PID', 'BangBang', 'DDPG', 'MPC'],
                        help='Control methods to simulate (default: Hysteresis)')
    
    # Sensor parameters
    parser.add_argument('--lidar-noise', type=float, default=0.05,
                        help='LiDAR measurement noise std deviation (default: 0.05)')
    parser.add_argument('--mpu-noise', type=float, default=0.1,
                        help='MPU6050 measurement noise std deviation (default: 0.1)')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Simulation time step in seconds (default: 0.01)')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Simulation duration in seconds (default: 20.0)')
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable automatic control (manual thrust only)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no visualization)')
    
    # Logging parameters
    parser.add_argument('--log', action='store_true',
                        help='Enable data logging')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Logging frequency (1 = every step, 10 = every 10th step)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to store log files')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze and plot results after simulation ends')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple controllers (requires --controllers with multiple values)')
    
    # DDPG specific parameters
    parser.add_argument('--ddpg-load', type=str, default=None,
                        help='Load pre-trained DDPG model from file')
    parser.add_argument('--ddpg-save', type=str, default=None,
                        help='Save trained DDPG model to file when simulation ends')
    parser.add_argument('--ddpg-no-train', action='store_true',
                        help='Disable DDPG training (evaluation mode only)')
    
    return parser.parse_args()


def main():
    """Main simulation function."""
    # Parse command line arguments
    args = parse_args = parse_arguments()
    
    # Create base configuration dictionary from args
    config = {
        'simulation': {
            'dt': args.dt,
            'log_freq': args.log_freq,
            'no_auto': args.no_auto,
            'duration': args.duration
        },
        'physics': {
            'mass': args.mass,
            'max_thrust': args.thrust,
            'delay_time': args.delay,
            'dt': args.dt
        },
        'sensors': {
            'lidar_noise_std': args.lidar_noise,
            'mpu_noise_std': args.mpu_noise,
            'lidar_update_rate': 10,  # 10 Hz
            'mpu_update_rate': 100    # 100 Hz
        },
        'kalman_filter': {
            'dt': args.dt,
            'measurement_noise_position': args.lidar_noise,
            'measurement_noise_acceleration': args.mpu_noise
        },
        'controllers': {
            'Hysteresis': {
                'target_height': args.target,
                'response_delay': args.delay/2,
                'dt': args.dt
            },
            'PID': {
                'target_height': args.target,
                'dt': args.dt
            },
            'BangBang': {
                'target_height': args.target,
                'dt': args.dt
            },
            'MPC': {
                'target_height': args.target,
                'response_delay': args.delay,
                'max_thrust': args.thrust,
                'mass': args.mass,
                'dt': args.dt
            },
            'DDPG': {
                'target_height': args.target,
                'dt': args.dt,
                'ddpg_load': args.ddpg_load,
                'ddpg_save': args.ddpg_save,
                'training_mode': not args.ddpg_no_train
            }
        }
    }
    
    # Create output directory if needed
    if args.log and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # Initialize shared components
    physics = PhysicsEngine(**config['physics'])
    sensors = SensorSystem(**config['sensors'])
    
    # Initialize visualization if not in headless mode
    visualizer = None if args.headless else Visualizer()
    
    # Check controller list
    if args.compare and len(args.controllers) < 2:
        print("Error: Comparison mode requires at least two controllers")
        return
    
    # Storage for simulation results
    results = {}
    
    # Run simulations for each specified controller
    for controller_name in args.controllers:
        print(f"\n--- Setting up {controller_name} Controller ---")
        
        # Create Kalman filter (reset for each controller)
        kalman = KalmanFilter(
            dt=config['kalman_filter']['dt'],
            initial_position=0.0,  # Will be reset before simulation
            measurement_noise_position=config['kalman_filter']['measurement_noise_position'],
            measurement_noise_acceleration=config['kalman_filter']['measurement_noise_acceleration']
        )
        
        # Fix controller name for BangBang to match factory method
        method_name = 'Bang-Bang' if controller_name == 'BangBang' else controller_name
        
        # Get controller-specific config
        controller_config = config['controllers'][controller_name]
        
        # Create controller using factory
        controller = create_controller(
            method=method_name,
            **controller_config
        )
        
        # Handle DDPG model loading if needed
        if controller_name == 'DDPG' and isinstance(controller, DDPGController):
            if args.ddpg_load and os.path.exists(args.ddpg_load):
                controller.load_networks(args.ddpg_load)
                print(f"Loaded pre-trained DDPG model from {args.ddpg_load}")
            controller.set_training_mode(not args.ddpg_no_train)
            if not args.ddpg_no_train:
                print("DDPG controller is in training mode")
            else:
                print("DDPG controller is in evaluation mode (no training)")
        
        # Initialize logger for this controller
        logger = SimulationLogger(log_folder=args.log_dir)
        
        # Create runner config
        runner_config = {
            **config['simulation'],
            'controller_name': controller_name,
            'initial_target': args.target,
            'step_target': args.target + 2.0  # Example of target change after 10s
        }
        
        # Create the simulation runner
        runner = SimulationRunner(
            physics=physics,
            sensors=sensors,
            kf=kalman,
            controller=controller,
            logger=logger,
            visualizer=visualizer,
            config=runner_config
        )
        
        # Run the simulation
        print(f"Running simulation with {controller_name} controller...")
        simulation_data = runner.run(args.duration)
        
        # Save DDPG model if requested
        if controller_name == 'DDPG' and args.ddpg_save and isinstance(controller, DDPGController):
            if not os.path.exists('models'):
                os.makedirs('models')
            save_path = args.ddpg_save
            controller.save_networks(save_path)
            print(f"Saved DDPG model to {save_path}")
        
        # Save data if logging is enabled
        if args.log:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}_{controller_name}.pkl"
            log_file = logger.save_data(filename)
            print(f"Saved simulation data to {log_file}")
        
        # Store results for comparison if needed
        if args.compare or args.analyze:
            results[controller_name] = simulation_data
        
        # If not comparing, show individual results immediately
        if args.analyze and not args.compare and not args.headless:
            plot_simulation_results(simulation_data, title=f"{controller_name} Controller")
            plot_kalman_performance(simulation_data)
            plot_controller_performance(simulation_data)
    
    # Plot comparison results if enabled
    if args.compare and args.analyze and len(results) > 1:
        print("\n--- Generating Comparison Plots ---")
        plot_simulation_results_comparison(results)
        plot_controller_performance_comparison(results)
    
    # Clean up
    if visualizer:
        visualizer.close()
    
    print("\nSimulation complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
