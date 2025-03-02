#!/usr/bin/env python3
"""
Pneumatic Hopper Simulation - Main Module

This program simulates a pneumatically actuated hopper constrained to vertical movement.
It implements a physics model with pneumatic delay, Kalman filtering for state estimation,
and a hysteresis controller for altitude control.

Run this script to start the simulation.
"""

import time
import numpy as np
import pygame
import argparse

from physics import PhysicsEngine
from sensor import SensorSystem
from kalman_filter import KalmanFilter
from controller import HysteresisController
from visualization import Visualizer


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
    parser.add_argument('--band', type=float, default=0.3,
                        help='Hysteresis band width in meters (default: 0.3)')
    
    # Sensor parameters
    parser.add_argument('--lidar-noise', type=float, default=0.05,
                        help='LiDAR measurement noise std deviation (default: 0.05)')
    parser.add_argument('--mpu-noise', type=float, default=0.1,
                        help='MPU6050 measurement noise std deviation (default: 0.1)')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Simulation time step in seconds (default: 0.01)')
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable automatic control (manual thrust only)')
    
    return parser.parse_args()


def main():
    """Main simulation function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create simulation components
    physics = PhysicsEngine(
        mass=args.mass,
        max_thrust=args.thrust,
        delay_time=args.delay,
        dt=args.dt
    )
    
    sensors = SensorSystem(
        lidar_noise_std=args.lidar_noise,
        mpu_noise_std=args.mpu_noise,
        lidar_update_rate=10,  # 10 Hz
        mpu_update_rate=100    # 100 Hz
    )
    
    kalman = KalmanFilter(
        dt=args.dt,
        initial_position=physics.position,
        measurement_noise_position=args.lidar_noise,
        measurement_noise_acceleration=args.mpu_noise
    )
    
    controller = HysteresisController(
        target_height=args.target,
        hysteresis_band=args.band,
        response_delay=args.delay/2,  # Controller response is half the pneumatic delay
        dt=args.dt
    )
    
    visualizer = Visualizer()
    
    # Initialize flags
    running = True
    manual_mode = args.no_auto
    manual_thrust = False
    
    # Main simulation loop
    while running:
        # Step the physics simulation
        physics.step()
        
        # Update sensors
        pos_reading, acc_reading, new_lidar, new_mpu = sensors.update(
            physics.simulation_time,
            physics.position,
            physics.acceleration
        )
        
        # Update Kalman filter with sensor readings
        kalman.step(pos_reading, acc_reading)
        est_pos, est_vel, est_acc = kalman.get_state()
        
        # Calculate control input
        if manual_mode:
            control_input = 1.0 if manual_thrust else 0.0
        else:
            # Use the controller with estimated state from Kalman filter
            control_input = controller.compute_control(est_pos, est_vel)
        
        # Apply control to physics
        physics.apply_control(control_input)
        
        # Update visualization
        events = visualizer.update(
            (physics.position, physics.velocity, physics.acceleration, physics.thrust / physics.max_thrust),
            (est_pos, est_vel, est_acc),
            {
                'target_height': controller.get_target_height(),
                'hysteresis_band': controller.hysteresis_band
            },
            (pos_reading, acc_reading, new_lidar, new_mpu)
        )
        
        # Handle user input
        actions = visualizer.handle_events(events)
        
        if actions['quit']:
            running = False
        
        if actions['reset']:
            physics.reset()
            sensors.reset()
            kalman.reset(physics.position)
            controller.reset()
            manual_thrust = False
        
        if actions['toggle_thrust']:
            if manual_mode:
                manual_thrust = not manual_thrust
            else:
                manual_mode = True
                manual_thrust = True
                print("Switched to manual control")
        
        if actions['adjust_target'] != 0.0:
            if manual_mode:
                manual_mode = False
                print("Switched to automatic control")
            controller.adjust_target_height(actions['adjust_target'])
    
    # Clean up
    visualizer.close()
    print("Simulation complete.")


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
