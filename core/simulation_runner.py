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

                # Handle Pause/Settings (Note: Disabling runtime parameter changes for comparison)
                if actions.get('toggle_settings', False):
                    self.simulation_paused = not self.simulation_paused
                    print(f"Simulation {'Paused' if self.simulation_paused else 'Resumed'}")
                    # We ignore the parameters.py/menu.py functionality for comparison runs

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
