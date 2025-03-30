#!/usr/bin/env python3
"""
Simulation Runner for the pneumatic hopper simulation.

This module contains the SimulationRunner class which encapsulates the main simulation
loop logic, handling physics updates, sensor readings, state estimation, control decisions,
and visualization.
"""

import pygame
import numpy as np
import time
from controllers.ddpg_controller import DDPGController  # Import needed for type checking


class SimulationRunner:
    def __init__(self, physics, sensors, kf, controller, logger, visualizer=None, config=None):
        """
        Initializes the simulation runner.

        Args:
            physics: Instance of PhysicsEngine
            sensors: Instance of SensorSystem
            kf: Instance of KalmanFilter
            controller: Instance of a controller (e.g., PIDController)
            logger: Instance of SimulationLogger (from core.logger)
            visualizer: Instance of Visualizer (optional)
            config: Dictionary containing simulation parameters (dt, controller_name etc.)
        """
        self.physics = physics
        self.sensors = sensors
        self.kf = kf
        self.controller = controller
        self.logger = logger
        self.visualizer = visualizer
        self.config = config if config else {}  # Store config

        # Extract needed params from config or use defaults
        self.dt = self.config.get('dt', 0.01)
        self.controller_name = self.config.get('controller_name', 'Unknown')
        self.log_freq = self.config.get('log_freq', 10)

        self.running = True
        self.manual_mode = self.config.get('no_auto', False)
        self.manual_thrust_on = False
        self.simulation_paused = False  # Add pause state

    def _get_target_height(self, current_time):
        """
        Get target height at a given time.
        
        This can be used to implement time-based target height profiles.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            float: Target height at the given time
        """
        # Simple target profile (replace with more complex logic if needed)
        # Example: step change at 10 seconds
        if current_time < 10.0:
            return self.config.get('initial_target', 3.0)
        else:
            return self.config.get('step_target', 5.0)

    def show_settings_dialog(self):
        """
        Display a settings dialog using direct Pygame rendering.
        
        Returns:
            tuple: (changes_applied, new_settings) where:
                - changes_applied is a boolean (True if applied, False if canceled)
                - new_settings is a dict with updated values (if applied)
        """
        if not self.visualizer:
            return False, None
        
        # Get screen object from visualizer
        screen = self.visualizer.screen
        
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
            'mass': self.physics.mass,
            'max_thrust': self.physics.max_thrust,
            'delay': self.physics.delay_time,
            'air_resistance': self.physics.air_resistance,
            'target_height': self.controller.get_target_height(),
            'lidar_detail': getattr(self.visualizer, 'show_lidar_detail', False)
        }
        
        # Create sliders for basic parameters
        sliders = {
            'mass': {'value': self.physics.mass, 'min': 0.1, 'max': 5.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 100, 200, 20)},
            'thrust': {'value': self.physics.max_thrust, 'min': 5.0, 'max': 50.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 150, 200, 20)},
            'delay': {'value': self.physics.delay_time, 'min': 0.0, 'max': 0.5, 'rect': pygame.Rect(panel_x + 200, panel_y + 200, 200, 20)},
            'air_res': {'value': self.physics.air_resistance, 'min': 0.0, 'max': 0.5, 'rect': pygame.Rect(panel_x + 200, panel_y + 250, 200, 20)},
            'target': {'value': self.controller.get_target_height(), 'min': 0.5, 'max': 10.0, 'rect': pygame.Rect(panel_x + 200, panel_y + 300, 200, 20)}
        }
        
        # Add MPC-specific sliders if the controller is MPC
        if self.controller_name == "MPC":
            # Get current values from the MPC controller
            prediction_horizon = getattr(self.controller, 'prediction_horizon', 15)
            
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
        if self.controller_name == "MPC":
            # Move checkbox lower for MPC since we have the horizon slider
            lidar_checkbox = pygame.Rect(panel_x + 200, panel_y + 380, 20, 20)
        else:
            lidar_checkbox = pygame.Rect(panel_x + 200, panel_y + 350, 20, 20)
        lidar_checked = getattr(self.visualizer, 'show_lidar_detail', False)
        
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
            if self.controller_name == "MPC" and 'horizon' in sliders:
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
            if self.controller_name == "MPC":
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
                            if self.controller_name == "MPC" and 'horizon' in sliders:
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

    def run(self, duration):
        """
        Runs the simulation for a given duration.
        
        Args:
            duration: Duration to run in seconds
            
        Returns:
            dict: Dictionary containing logged simulation data
        """
        num_steps = int(duration / self.dt)
        step_counter = 0

        # Reset components before run
        initial_target = self._get_target_height(0.0)
        self.physics.reset()
        self.sensors.reset()
        self.kf.reset(self.physics.position)
        self.controller.reset(target_height=initial_target)
        self.logger.reset()

        self.running = True
        print(f"\n--- Running Simulation: Controller = {self.controller_name} ---")

        while self.running and step_counter < num_steps:
            current_time = step_counter * self.dt

            # --- Event Handling (if visualizer exists) ---
            if self.visualizer:
                time_delta = self.visualizer.clock.get_time() / 1000.0  # For UI updates if any
                events = pygame.event.get()
                actions = self.visualizer.handle_events(events)  # Delegate most event handling

                if actions['quit']:
                    self.running = False
                    break
                if actions['reset']:
                    # Reset logic remains here as it affects the whole runner state
                    initial_target = self._get_target_height(0.0)
                    self.physics.reset()
                    self.sensors.reset()
                    self.kf.reset(self.physics.position)
                    self.controller.reset(target_height=initial_target)
                    self.logger.reset()  # Also reset logger
                    self.manual_thrust_on = False
                    self.manual_mode = self.config.get('no_auto', False)
                    self.simulation_paused = False  # Ensure unpaused on reset
                    print("Simulation reset.")
                    step_counter = 0  # Restart time
                    current_time = 0.0
                    continue  # Skip rest of loop for this iteration

                # Handle manual mode toggle (simplified)
                if actions['toggle_thrust']:
                    if self.manual_mode:
                        self.manual_thrust_on = not self.manual_thrust_on
                    else:
                        self.manual_mode = True
                        self.manual_thrust_on = True

                # Handle target adjustment (simplified)
                if actions['adjust_target'] != 0.0:
                    if self.manual_mode:
                        self.manual_mode = False  # Switch back to auto on target adjust
                    new_target = self.controller.get_target_height() + actions['adjust_target']
                    self.controller.set_target_height(new_target)
                    # We are no longer directly changing target based on time profile
                    # We'll use the controller's internal target state now

                # Handle Pause/Settings
                if actions.get('toggle_settings', False):
                    self.simulation_paused = True  # Pause the simulation
                    print(f"Simulation paused for settings adjustment")
                    
                    # Save current parameter values to restore if needed
                    old_mass = self.physics.mass
                    old_max_thrust = self.physics.max_thrust
                    old_delay = self.physics.delay_time
                    old_air_resistance = self.physics.air_resistance
                    old_target = self.controller.get_target_height()
                    
                    # Show the settings dialog
                    changes_applied, new_settings = self.show_settings_dialog()
                    
                    if changes_applied and new_settings:
                        # Apply the new settings
                        self.physics.mass = new_settings['mass']
                        self.physics.max_thrust = new_settings['max_thrust']
                        self.physics.delay_time = new_settings['delay_time']
                        self.physics.delay_steps = int(self.physics.delay_time / self.physics.dt)
                        self.physics.control_history = [0.0] * self.physics.delay_steps
                        self.physics.air_resistance = new_settings['air_resistance']
                        
                        # Apply target height
                        if hasattr(self.controller, 'set_target_height'):
                            self.controller.set_target_height(new_settings['target_height'])
                            
                        # Apply MPC-specific settings
                        if self.controller_name == 'MPC' and 'prediction_horizon' in new_settings:
                            if hasattr(self.controller, 'prediction_horizon'):
                                self.controller.prediction_horizon = new_settings['prediction_horizon']
                                # Reset controller's initial guess to match new horizon
                                self.controller.initial_guess = np.ones(self.controller.prediction_horizon) * 0.5
                                self.controller.last_solution = None
                                print(f"Updated MPC prediction horizon to {self.controller.prediction_horizon}")
                        
                        # Update visualizer settings
                        if hasattr(self.visualizer, 'show_lidar_detail'):
                            self.visualizer.show_lidar_detail = new_settings['lidar_detail']
                        
                        print("Applied new settings to simulation")
                        
                        # Reset after applying settings to ensure proper simulation restart
                        self.physics.reset()
                        self.sensors.reset()
                        self.kf.reset(self.physics.position)
                        self.controller.reset()
                        self.manual_thrust_on = False
                        print("Simulation reset with new settings")
                    else:
                        # User canceled, restore old values if needed
                        self.physics.mass = old_mass
                        self.physics.max_thrust = old_max_thrust
                        self.physics.delay_time = old_delay
                        self.physics.air_resistance = old_air_resistance
                        if hasattr(self.controller, 'set_target_height'):
                            self.controller.set_target_height(old_target)
                    
                    # Resume simulation after settings dialog is closed
                    self.simulation_paused = False
                    print("Simulation resumed")

            # --- Simulation Step (only if not paused) ---
            if not self.simulation_paused:
                # 1. Sensing
                pos_reading, acc_reading, new_lidar, new_mpu = self.sensors.update(
                    current_time, self.physics.position, self.physics.acceleration
                )

                # 2. Estimation
                # Predict based on the *last command* sent to physics, not current output
                # Note: Kalman predict doesn't need action if using simple model
                self.kf.predict()
                self.kf.update_position(pos_reading)
                self.kf.update_acceleration(acc_reading)
                est_pos, est_vel, est_acc = self.kf.get_state()

                # 3. Control Decision
                # Use the target height stored within the controller
                current_target_height = self.controller.get_target_height()

                if self.manual_mode:
                    control_output = 1.0 if self.manual_thrust_on else 0.0
                else:
                    # Pass estimated state to controller
                    if hasattr(self.controller, 'compute_control') and callable(getattr(self.controller, 'compute_control')):
                        if isinstance(self.controller, DDPGController):
                            control_output = self.controller.compute_control(est_pos, est_vel, est_acc)
                        else:
                            control_output = self.controller.compute_control(est_pos, est_vel)
                    else:
                        # Fallback if compute_control not defined
                        control_output = 0.0

                # Handle DDPG reward/learning step if applicable
                if isinstance(self.controller, DDPGController) and hasattr(self.controller, 'training_mode'):
                    if self.controller.training_mode and hasattr(self.controller, 'provide_reward'):
                        # Pass necessary info for reward calculation and learning step
                        self.controller.provide_reward(est_pos, est_vel)

                # 4. Physics Update (Apply Control to physics engine)
                self.physics.apply_control(control_output)
                self.physics.step()  # Physics advances

                # 5. Logging (conditional)
                if step_counter % self.log_freq == 0:
                    log_data = {
                        'time': current_time,
                        'true_position': self.physics.position, 
                        'true_velocity': self.physics.velocity, 
                        'true_acceleration': self.physics.acceleration,
                        'est_position': est_pos, 
                        'est_velocity': est_vel, 
                        'est_acceleration': est_acc,
                        'lidar_reading': pos_reading, 
                        'mpu_reading': acc_reading,
                        'target_height': current_target_height,
                        'control_output': control_output,  # The command decided this step
                        'applied_thrust': self.physics.thrust,  # The thrust actually applied (delayed)
                        'kf_covariance_diag': np.diag(self.kf.P) if hasattr(self.kf, 'P') else [np.nan]*3
                    }
                    # Add DDPG reward if applicable
                    if isinstance(self.controller, DDPGController) and hasattr(self.controller, 'last_reward'):
                        log_data['reward'] = self.controller.last_reward
                    self.logger.log_step(log_data)

                step_counter += 1  # Increment step counter only if not paused

            # --- Visualization Update (Always update display, even if paused) ---
            if self.visualizer:
                vis_physics_state = (self.physics.position, self.physics.velocity, self.physics.acceleration, self.physics.thrust / self.physics.max_thrust)
                vis_kalman_state = self.kf.get_state()  # Use current KF state for display
                vis_controller_data = {
                    'target_height': self.controller.get_target_height(),
                    'hysteresis_band': getattr(self.controller, 'hysteresis_band', 0.0)  # Get band if exists
                }
                # Get latest readings even if not new this step
                latest_lidar, latest_mpu = self.sensors.get_latest_readings()
                vis_sensor_data = (latest_lidar, latest_mpu, new_lidar, new_mpu)  # Still pass flags

                self.visualizer.update(vis_physics_state, vis_kalman_state, vis_controller_data, vis_sensor_data)
                self.visualizer.clock.tick(60)  # Limit FPS

        print(f"--- Simulation Finished: Controller = {self.controller_name} ---")
        return self.logger.get_data()  # Return logged data
