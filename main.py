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
import pygame_gui
import argparse
import os

from physics import PhysicsEngine
from sensor import SensorSystem
from kalman_filter import KalmanFilter
from visualization import Visualizer
from parameters import ParameterPanel
from analysis import SimulationLogger, plot_simulation_results, plot_kalman_performance, plot_controller_performance

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
    # Fix: Changed 'Bang-Bang' to 'BangBang' to avoid hyphen issues in command-line arguments
    parser.add_argument('--control', type=str, default='Hysteresis',
                        choices=['Hysteresis', 'PID', 'BangBang', 'DDPG'],
                        help='Control method (default: Hysteresis)')
    
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
    parser.add_argument('--no-params', action='store_true',
                        help='Hide parameter adjustment panel')
    
    # Logging parameters
    parser.add_argument('--log', action='store_true',
                        help='Enable data logging')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Logging frequency (1 = every step, 10 = every 10th step)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to store log files')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze and plot results after simulation ends')
    
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
    
    # Fix: Handle the case where 'BangBang' is provided in command line but needs to be 'Bang-Bang' for the factory
    control_method = args.control
    if control_method == 'BangBang':
        control_method = 'Bang-Bang'
    
    # Create initial controller using factory function
    controller = create_controller(
        method=control_method,
        target_height=args.target,
        response_delay=args.delay/2,  # For hysteresis controller
        dt=args.dt
    )
    
    # Load pre-trained DDPG model if specified and applicable
    if control_method == 'DDPG' and args.ddpg_load:
        if os.path.exists(args.ddpg_load):
            controller.load_networks(args.ddpg_load)
            print(f"Loaded pre-trained DDPG model from {args.ddpg_load}")
        else:
            print(f"Warning: Specified DDPG model file {args.ddpg_load} not found, using default initialization")
    
    # Set DDPG controller training mode based on argument
    if control_method == 'DDPG' and args.ddpg_no_train:
        controller.set_training_mode(False)
        print("DDPG controller is in evaluation mode (no training)")
    
    # Set up visualization
    visualizer = Visualizer()
    
    # Set up parameter panel if enabled
    param_panel = None
    param_panel_visible = False
    if not args.no_params:
        param_panel = ParameterPanel(400, 600, physics, controller, lidar_detail_enabled=visualizer.show_lidar_detail)
        param_panel.set_visible(param_panel_visible)
    
    # Initialize data logger if needed
    logger = None
    if args.log:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        logger = SimulationLogger(log_folder=args.log_dir)
    
    # Initialize flags
    running = True
    manual_mode = args.no_auto
    manual_thrust = False
    step_counter = 0
    current_control_method = control_method  # Fix: Use the corrected method name
    
    # Main simulation loop
    while running:
        # Get time delta for UI updates
        time_delta = visualizer.clock.get_time() / 1000.0
        
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
            if current_control_method == 'DDPG':
                control_input = controller.compute_control(est_pos, est_vel, est_acc)
                # Provide reward for DDPG learning after control action is computed
                controller.provide_reward(est_pos, est_vel)
            else:
                control_input = controller.compute_control(est_pos, est_vel)
        
        # Apply control to physics
        physics.apply_control(control_input)
        
        # Log data if enabled (with frequency control)
        if logger and step_counter % args.log_freq == 0:
            logger.log_physics(
                physics.simulation_time,
                physics.position,
                physics.velocity,
                physics.acceleration,
                physics.thrust
            )
            
            logger.log_kalman(
                physics.simulation_time,
                est_pos,
                est_vel,
                est_acc,
                np.diag(kalman.P)
            )
            
            logger.log_controller(
                physics.simulation_time,
                controller.get_target_height(),
                controller.get_target_height() - est_pos,
                control_input
            )
            
            logger.log_sensors(
                physics.simulation_time,
                pos_reading,
                acc_reading
            )
            
            # Additional logging for DDPG rewards if applicable
            if current_control_method == 'DDPG' and hasattr(controller, 'reward_history') and controller.reward_history:
                # Get the most recent reward
                recent_reward = controller.reward_history[-1] if controller.reward_history else 0
                # Log as additional data
                logger.log_additional(
                    physics.simulation_time,
                    {'reward': recent_reward}
                )
        
        # Update visualization
        events = visualizer.update(
            (physics.position, physics.velocity, physics.acceleration, physics.thrust / physics.max_thrust),
            (est_pos, est_vel, est_acc),
            {
                'target_height': controller.get_target_height(),
                'hysteresis_band': getattr(controller, 'hysteresis_band', 0.3)
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
        
        if actions['toggle_lidar']:
            visualizer.show_lidar_detail = not visualizer.show_lidar_detail
            if param_panel:
                param_panel.lidar_detail_enabled = visualizer.show_lidar_detail
                param_panel._update_checkbox_states()
        
        # Handle settings button toggle
        if actions['toggle_settings']:
            param_panel_visible = not param_panel_visible
            if param_panel:
                param_panel.set_visible(param_panel_visible)
        
        # Handle parameter panel events if visible
        if param_panel and param_panel_visible:
            param_panel.update(time_delta)
            
            # Process events through parameter panel
            for event in events:
                param_changes = param_panel.handle_event(event)
                
                # Check if controller method changed
                if param_changes['control_method_changed']:
                    controller_params = param_panel.get_controller_parameters()
                    new_method = controller_params['method']
                    
                    if new_method != current_control_method:
                        print(f"Switching control method from {current_control_method} to {new_method}")
                        
                        # Save DDPG model if switching away from it and save path specified
                        if current_control_method == 'DDPG' and args.ddpg_save and isinstance(controller, DDPGController):
                            ddpg_save_path = args.ddpg_save
                            controller.save_networks(ddpg_save_path)
                            print(f"Saved DDPG model to {ddpg_save_path}")
                        
                        current_control_method = new_method
                        
                        # Create new controller with current target height
                        current_target = controller.get_target_height()
                        controller = create_controller(
                            method=new_method,
                            target_height=current_target,
                            response_delay=physics.delay_time/2,
                            dt=args.dt,
                            **controller_params
                        )
                        
                        # Load DDPG model if switching to it and load path specified
                        if new_method == 'DDPG' and args.ddpg_load and isinstance(controller, DDPGController):
                            if os.path.exists(args.ddpg_load):
                                controller.load_networks(args.ddpg_load)
                                print(f"Loaded pre-trained DDPG model from {args.ddpg_load}")
                            
                        # Set DDPG training mode based on argument
                        if new_method == 'DDPG' and args.ddpg_no_train and isinstance(controller, DDPGController):
                            controller.set_training_mode(False)
                            print("DDPG controller is in evaluation mode (no training)")
                
                # Apply changes if requested
                if param_changes['apply_changes']:
                    # Reset simulation state
                    physics.reset()
                    sensors.reset()
                    kalman.reset(physics.position)
                    controller.reset(controller.get_target_height())
                    manual_thrust = False
                    print("Applied parameter changes and reset simulation state.")
                
                # Update LiDAR visualization if changed
                if param_changes['lidar_detail_changed']:
                    visualizer.show_lidar_detail = param_panel.get_lidar_detail_enabled()
                
                # Reset parameters if requested
                if param_changes['reset_params']:
                    param_panel.reset_parameters()
            
            # Draw parameter panel
            param_panel.draw(visualizer.screen)
            
            # Update display after drawing parameter panel
            pygame.display.flip()
        
        # Increment step counter
        step_counter += 1
    
    # Save DDPG model if requested before exiting
    if current_control_method == 'DDPG' and args.ddpg_save and isinstance(controller, DDPGController):
        ddpg_save_path = args.ddpg_save
        controller.save_networks(ddpg_save_path)
        print(f"Saved DDPG model to {ddpg_save_path}")
    
    # Save log data if enabled
    log_file = None
    if logger:
        log_file = logger.save_data()
        
        # Analyze and plot results if requested
        if args.analyze and log_file:
            data = SimulationLogger.load_data(log_file)
            plot_simulation_results(data)
            plot_kalman_performance(data)
            plot_controller_performance(data)
            
            # Add DDPG specific plots if applicable
            if current_control_method == 'DDPG' and 'additional' in data and 'reward' in data['additional']:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.plot(data['additional']['time'], data['additional']['reward'])
                    plt.title('DDPG Learning Curve')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Reward')
                    plt.grid(True)
                    plt.savefig(os.path.join(args.log_dir, 'ddpg_learning_curve.png'))
                    plt.close()
                    print(f"DDPG learning curve saved to {os.path.join(args.log_dir, 'ddpg_learning_curve.png')}")
                except Exception as e:
                    print(f"Error generating DDPG learning curve: {e}")
    
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
