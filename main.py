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
# parameters.py no longer used as we've replaced it with direct pygame UI
from analysis import SimulationLogger, plot_simulation_results, plot_kalman_performance, plot_controller_performance

def show_simple_settings_dialog(screen, physics, controller, lidar_detail, controller_type="Default"):
    """
    Display a simple settings dialog using direct Pygame rendering.
    
    Args:
        screen: Pygame screen to draw on
        physics: Physics engine instance
        controller: Current controller instance
        lidar_detail: Whether to show enhanced LiDAR visualization
        controller_type: Type of controller (for specialized settings)
    
    Returns:
        tuple: (changes_applied, new_settings) where:
            - changes_applied is a boolean (True if applied, False if canceled)
            - new_settings is a dict with updated values (if applied)
    """
    # Get screen dimensions
    screen_width, screen_height = screen.get_size()
    
    # Settings panel dimensions
    panel_width = 500
    panel_height = 500
    panel_x = (screen_width - panel_width) // 2
    panel_y = (screen_height - panel_height) // 2
    
    # Create font and colors
    font_large = pygame.font.SysFont('Arial', 24, bold=True)
    font = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 16)
    
    bg_color = (240, 240, 240)
    title_color = (50, 50, 150)
    text_color = (0, 0, 0)
    slider_color = (180, 180, 180)
    slider_handle_color = (100, 100, 100)
    button_color = (200, 200, 200)
    button_hover_color = (220, 220, 220)
    
    # Save initial settings for potential cancellation
    current_settings = {
        'mass': physics.mass,
        'max_thrust': physics.max_thrust,
        'delay': physics.delay_time,
        'air_resistance': physics.air_resistance,
        'target_height': controller.get_target_height(),
        'lidar_detail': lidar_detail
    }
    
    # Create sliders for basic parameters
    sliders = {
        'mass': {'value': physics.mass, 'min': 0.1, 'max': 5.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 100, 200, 20)},
        'thrust': {'value': physics.max_thrust, 'min': 5.0, 'max': 50.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 150, 200, 20)},
        'delay': {'value': physics.delay_time, 'min': 0.0, 'max': 0.5, 'rect': pygame.Rect(panel_x + 200, panel_y + 200, 200, 20)},
        'air_res': {'value': physics.air_resistance, 'min': 0.0, 'max': 0.5, 'rect': pygame.Rect(panel_x + 200, panel_y + 250, 200, 20)},
        'target': {'value': controller.get_target_height(), 'min': 0.5, 'max': 10.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 300, 200, 20)}
    }
    
    # Add MPC-specific sliders if the controller is MPC
    if controller_type == "MPC":
        # Get current values from the MPC controller
        prediction_horizon = getattr(controller, 'prediction_horizon', 15)
        
        # Add MPC-specific sliders
        sliders['horizon'] = {
            'value': prediction_horizon, 
            'min': 5, 
            'max': 30, 
            'rect': pygame.Rect(panel_x + 200, panel_y + 350, 200, 20)
        }
    
    # Create buttons
    apply_button = pygame.Rect(panel_x + 150, panel_y + 400, 100, 40)
    cancel_button = pygame.Rect(panel_x + 270, panel_y + 400, 100, 40)
    
    # Create checkbox for LiDAR detail - position depends on controller type
    if controller_type == "MPC":
        # Move checkbox lower for MPC since we have the horizon slider
        lidar_checkbox = pygame.Rect(panel_x + 200, panel_y + 380, 20, 20)
    else:
        lidar_checkbox = pygame.Rect(panel_x + 200, panel_y + 350, 20, 20)
    lidar_checked = lidar_detail
    
    # Main dialog loop
    dialog_running = True
    dragging_slider = None
    changes_applied = False
    new_settings = None  # Initialize new_settings
    
    while dialog_running:
        # Create a semi-transparent overlay for background dimming
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        screen.blit(overlay, (0, 0))
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(screen, bg_color, panel_rect, border_radius=10)
        pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2, border_radius=10)
        
        # Draw title
        title = font_large.render("Simulation Settings", True, title_color)
        screen.blit(title, (panel_x + (panel_width - title.get_width()) // 2, panel_y + 30))
        
        # Draw sliders and labels
        mass_label = font.render(f"Mass: {sliders['mass']['value']:.2f} kg", True, text_color)
        screen.blit(mass_label, (panel_x + 50, panel_y + 100))
        
        thrust_label = font.render(f"Max Thrust: {sliders['thrust']['value']:.1f} N", True, text_color)
        screen.blit(thrust_label, (panel_x + 50, panel_y + 150))
        
        delay_label = font.render(f"Delay: {sliders['delay']['value']:.2f} s", True, text_color)
        screen.blit(delay_label, (panel_x + 50, panel_y + 200))
        
        air_res_label = font.render(f"Air Resistance: {sliders['air_res']['value']:.2f}", True, text_color)
        screen.blit(air_res_label, (panel_x + 50, panel_y + 250))
        
        target_label = font.render(f"Target Height: {sliders['target']['value']:.2f} m", True, text_color)
        screen.blit(target_label, (panel_x + 50, panel_y + 300))
        
        # Draw MPC-specific labels if applicable
        if controller_type == "MPC" and 'horizon' in sliders:
            horizon_label = font.render(f"Prediction Horizon: {int(sliders['horizon']['value'])}", True, text_color)
            screen.blit(horizon_label, (panel_x + 50, panel_y + 350))
        
        # Draw sliders
        for key, slider in sliders.items():
            # Draw slider track
            pygame.draw.rect(screen, slider_color, slider['rect'])
            
            # Calculate handle position
            value_ratio = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
            handle_x = slider['rect'].left + int(value_ratio * slider['rect'].width)
            handle_rect = pygame.Rect(handle_x - 5, slider['rect'].top - 5, 10, slider['rect'].height + 10)
            
            # Draw handle
            pygame.draw.rect(screen, slider_handle_color, handle_rect, border_radius=5)
        
        # Draw LiDAR checkbox
        pygame.draw.rect(screen, (255, 255, 255), lidar_checkbox)
        pygame.draw.rect(screen, (0, 0, 0), lidar_checkbox, 1)
        if lidar_checked:
            # Draw X mark
            pygame.draw.line(screen, (0, 0, 0), 
                            (lidar_checkbox.left + 3, lidar_checkbox.top + 3),
                            (lidar_checkbox.right - 3, lidar_checkbox.bottom - 3), 2)
            pygame.draw.line(screen, (0, 0, 0), 
                            (lidar_checkbox.left + 3, lidar_checkbox.bottom - 3),
                            (lidar_checkbox.right - 3, lidar_checkbox.top + 3), 2)
        
        # Position the LiDAR label correctly based on controller type
        lidar_label = font.render("Show Enhanced LiDAR", True, text_color)
        if controller_type == "MPC":
            screen.blit(lidar_label, (panel_x + 230, panel_y + 380))
        else:
            screen.blit(lidar_label, (panel_x + 230, panel_y + 350))
        
        # Draw buttons
        pygame.draw.rect(screen, button_color, apply_button, border_radius=5)
        pygame.draw.rect(screen, (0, 0, 0), apply_button, 1, border_radius=5)
        apply_text = font.render("Apply", True, text_color)
        screen.blit(apply_text, (apply_button.x + (apply_button.width - apply_text.get_width()) // 2,
                               apply_button.y + (apply_button.height - apply_text.get_height()) // 2))
        
        pygame.draw.rect(screen, button_color, cancel_button, border_radius=5)
        pygame.draw.rect(screen, (0, 0, 0), cancel_button, 1, border_radius=5)
        cancel_text = font.render("Cancel", True, text_color)
        screen.blit(cancel_text, (cancel_button.x + (cancel_button.width - cancel_text.get_width()) // 2,
                                cancel_button.y + (cancel_button.height - cancel_text.get_height()) // 2))
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dialog_running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check if clicked on slider handle
                    for key, slider in sliders.items():
                        value_ratio = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
                        handle_x = slider['rect'].left + int(value_ratio * slider['rect'].width)
                        handle_rect = pygame.Rect(handle_x - 10, slider['rect'].top - 10, 20, slider['rect'].height + 20)
                        
                        if handle_rect.collidepoint(mouse_pos):
                            dragging_slider = key
                            break
                    
                    # Check if clicked on checkbox
                    if lidar_checkbox.collidepoint(mouse_pos):
                        lidar_checked = not lidar_checked
                    
                    # Check if clicked on buttons
                    if apply_button.collidepoint(mouse_pos):
                        # Store the base settings
                        new_settings = {
                            'mass': sliders['mass']['value'],
                            'max_thrust': sliders['thrust']['value'],
                            'delay_time': sliders['delay']['value'],
                            'air_resistance': sliders['air_res']['value'],
                            'target_height': sliders['target']['value'],
                            'lidar_detail': lidar_checked
                        }
                        
                        # Add MPC-specific settings if applicable
                        if controller_type == "MPC" and 'horizon' in sliders:
                            new_settings['prediction_horizon'] = int(sliders['horizon']['value'])
                        
                        changes_applied = True
                        dialog_running = False
                        
                    if cancel_button.collidepoint(mouse_pos):
                        # Cancel and close dialog
                        dialog_running = False
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging_slider = None
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging_slider:
                    mouse_x = pygame.mouse.get_pos()[0]
                    slider = sliders[dragging_slider]
                    
                    # Calculate new slider value based on mouse position
                    value_ratio = max(0, min(1, (mouse_x - slider['rect'].left) / slider['rect'].width))
                    slider['value'] = slider['min'] + value_ratio * (slider['max'] - slider['min'])
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # ESC key closes dialog without applying changes
                    dialog_running = False
        
        # Update display
        pygame.display.flip()
        pygame.time.delay(10)  # Small delay to reduce CPU usage
    
    # Return both the status and the new settings (or None if canceled)
    return (changes_applied, new_settings if changes_applied else None)

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
                        choices=['Hysteresis', 'PID', 'BangBang', 'DDPG', 'MPC'],
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

    # DDPG Training
    parser.add_argument('--headless', action='store_true',
                    help='Run in headless mode (no visualization) for faster training')
    parser.add_argument('--training-steps', type=int, default=1000000,
                    help='Number of simulation steps to run in headless mode')
    parser.add_argument('--save-interval', type=int, default=50000,
                    help='Save checkpoints every N steps in headless mode')
    
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
    visualizer = None
    if not args.headless:
        visualizer = Visualizer()
    
    # We're no longer using the parameter panel
    
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
    max_steps = 1000000
    current_control_method = control_method  # Fix: Use the corrected method name
    simulation_paused = False  # Add pause flag

    report_interval = 10000
    
    # Main simulation loop
    while running and (not args.headless or step_counter < max_steps):
        # Get time delta for UI updates
        time_delta = visualizer.clock.get_time() / 1000.0
        
        # Only update physics when not paused
        if not simulation_paused:
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
        else:
            # When paused, just get the current state values without updating them
            est_pos, est_vel, est_acc = kalman.get_state()
            pos_reading, acc_reading = sensors.get_latest_readings()
            new_lidar, new_mpu = False, False
        
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
        if not args.headless:
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
                # Reset all simulation components
                physics.reset()
                sensors.reset()
                kalman.reset(physics.position)
                controller.reset()
                manual_thrust = False
                
                # Ensure simulation is running
                simulation_paused = False
                
                # Print confirmation
                print("Simulation reset and resumed")
                manual_mode = args.no_auto  # Reset to default control mode
                simulation_paused = False   # Ensure simulation isn't paused
                print("Simulation reset - control system restarted")
            
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
            
            # Handle settings button toggle
            if actions['toggle_settings']:
                # Pause the simulation
                simulation_paused = True
                
                # Save current parameter values to restore if needed
                old_mass = physics.mass
                old_max_thrust = physics.max_thrust
                old_delay = physics.delay_time
                old_air_resistance = physics.air_resistance
                old_target = controller.get_target_height()
                
                # Show a very simple dialog using Pygame directly
                if not args.headless and visualizer:
                    # Determine controller type for appropriate settings dialog
                    controller_type = "Default"
                    if current_control_method == 'MPC':
                        controller_type = "MPC"
                    
                    # Use a function to display and handle a simple settings menu
                    changes_applied, new_settings = show_simple_settings_dialog(
                        visualizer.screen, 
                        physics, 
                        controller, 
                        visualizer.show_lidar_detail,
                        controller_type
                    )
                    
                    if changes_applied and new_settings:
                        # Apply the new settings
                        physics.mass = new_settings['mass']
                        physics.max_thrust = new_settings['max_thrust']
                        physics.delay_time = new_settings['delay_time']
                        physics.delay_steps = int(physics.delay_time / physics.dt)
                        physics.control_history = [0.0] * physics.delay_steps
                        physics.air_resistance = new_settings['air_resistance']
                        
                        # Apply target height
                        if hasattr(controller, 'set_target_height'):
                            controller.set_target_height(new_settings['target_height'])
                            
                        # Apply MPC-specific settings
                        if current_control_method == 'MPC' and 'prediction_horizon' in new_settings:
                            if hasattr(controller, 'prediction_horizon'):
                                controller.prediction_horizon = new_settings['prediction_horizon']
                                # Reset controller's initial guess to match new horizon
                                controller.initial_guess = np.ones(controller.prediction_horizon) * 0.5
                                controller.last_solution = None
                                print(f"Updated MPC prediction horizon to {controller.prediction_horizon}")
                                
                        visualizer.show_lidar_detail = new_settings['lidar_detail']
                        print("Applied new settings to simulation")
                        
                        # Reset after applying settings to ensure proper simulation restart
                        physics.reset()
                        sensors.reset()
                        kalman.reset(physics.position)
                        controller.reset()
                        manual_thrust = False
                        print("Simulation reset with new settings")
                    else:
                        # User canceled, restore old values if needed
                        physics.mass = old_mass
                        physics.max_thrust = old_max_thrust
                        physics.delay_time = old_delay
                        physics.air_resistance = old_air_resistance
                        if hasattr(controller, 'set_target_height'):
                            controller.set_target_height(old_target)
                
                # Unpause and make sure the controller is in the right mode
                simulation_paused = False
                manual_mode = args.no_auto  # Reset to command line default
                
                # Give the controller a kick-start with an initial control computation
                est_pos, est_vel, est_acc = kalman.get_state()
                if not manual_mode:
                    # Compute an initial control action to get things moving
                    if current_control_method == 'DDPG':
                        control_input = controller.compute_control(est_pos, est_vel, est_acc)
                    else:
                        control_input = controller.compute_control(est_pos, est_vel)
                        
                    # Apply this initial control to physics
                    physics.apply_control(control_input)
                    
                print("Simulation resumed with control system active")
                control_input = 0.0
                if not manual_mode:
                    # Force the controller to compute an initial control value
                    est_pos, est_vel, est_acc = kalman.get_state()
                    if current_control_method == 'DDPG':
                        control_input = controller.compute_control(est_pos, est_vel, est_acc)
                    else:
                        control_input = controller.compute_control(est_pos, est_vel)
                    physics.apply_control(control_input)
        
            # All parameter panel handling is now done in the show_simple_settings_dialog function
        else:
            # For headless mode, provide some progress feedback
            if step_counter % report_interval == 0:
                avg_reward = sum(controller.reward_history[-report_interval:]) / min(report_interval, len(controller.reward_history))
                error = controller.get_target_height() - est_pos
                print(f"Step {step_counter}: Pos={est_pos:.2f}, Error={error:.2f}, Avg Reward={avg_reward:.4f}")
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

    if args.ddpg_save:
        controller.save_networks(args.ddpg_save)
        print(f"Saved trained model to {args.ddpg_save}")
    
    # Clean up
    if visualizer:
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
