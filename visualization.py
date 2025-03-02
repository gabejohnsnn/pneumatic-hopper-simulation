import pygame
import numpy as np
import time
from pygame import gfxdraw

class Visualizer:
    """
    Pygame-based visualizer for the pneumatic hopper simulation.
    """
    
    def __init__(self, width=1200, height=800, scale_factor=100, ground_height=50):
        """
        Initialize the visualizer.
        
        Args:
            width (int): Screen width in pixels
            height (int): Screen height in pixels
            scale_factor (float): Pixels per meter for world-to-screen conversion
            ground_height (int): Height of the ground in pixels from bottom
        """
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up the screen
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pneumatic Hopper Simulation")
        
        # Set up fonts
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Scaling and coordinate system
        self.scale_factor = scale_factor
        self.ground_height = ground_height
        
        # Colors
        self.bg_color = (240, 240, 240)
        self.ground_color = (100, 100, 100)
        self.track_color = (180, 180, 180)
        self.hopper_color = (50, 100, 200)
        self.thrust_color = (255, 165, 0)
        self.text_color = (20, 20, 20)
        self.target_color = (255, 0, 0, 100)  # Red with alpha
        self.hysteresis_upper_color = (255, 100, 100, 50)  # Lighter red with alpha
        self.hysteresis_lower_color = (100, 100, 255, 50)  # Lighter blue with alpha
        
        # Control panel
        self.control_panel_width = 400
        self.control_panel_rect = pygame.Rect(width - self.control_panel_width, 0, 
                                             self.control_panel_width, height)
        
        # Time tracking
        self.last_update_time = time.time()
        self.frame_times = []
        self.max_frame_times = 100
        
        # Visualization flags
        self.show_kalman = True
        self.show_hysteresis = True
        
        # Clock for limiting frame rate
        self.clock = pygame.time.Clock()
    
    def world_to_screen_y(self, world_y):
        """
        Convert from world y coordinate (meters) to screen y coordinate (pixels).
        In world coordinates, y increases upward. In screen coordinates, y increases downward.
        
        Args:
            world_y (float): Y coordinate in world space (meters)
            
        Returns:
            int: Y coordinate in screen space (pixels)
        """
        return int(self.height - self.ground_height - world_y * self.scale_factor)
    
    def draw_background(self):
        """Draw the background, ground, and track."""
        # Clear the screen
        self.screen.fill(self.bg_color)
        
        # Draw ground
        ground_rect = pygame.Rect(0, self.height - self.ground_height, 
                                 self.width - self.control_panel_width, self.ground_height)
        pygame.draw.rect(self.screen, self.ground_color, ground_rect)
        
        # Draw track
        track_width = 30
        track_x = (self.width - self.control_panel_width) // 2 - track_width // 2
        track_rect = pygame.Rect(track_x, 0, track_width, self.height - self.ground_height)
        pygame.draw.rect(self.screen, self.track_color, track_rect)
    
    def draw_hopper(self, position, velocity, thrust_level):
        """
        Draw the hopper at the given position.
        
        Args:
            position (float): Hopper position (height) in meters
            velocity (float): Hopper velocity in m/s
            thrust_level (float): Current thrust level (0 to 1)
        """
        # Calculate screen coordinates
        hopper_radius = 25
        hopper_x = (self.width - self.control_panel_width) // 2
        hopper_y = self.world_to_screen_y(position)
        
        # Draw the hopper body (circle)
        pygame.draw.circle(self.screen, self.hopper_color, (hopper_x, hopper_y), hopper_radius)
        
        # Draw the thruster flame if active
        if thrust_level > 0:
            # Flame size based on thrust level
            flame_length = int(30 * thrust_level)
            
            # Draw flame as a triangle
            flame_points = [
                (hopper_x, hopper_y + hopper_radius),
                (hopper_x - 15, hopper_y + hopper_radius + flame_length),
                (hopper_x + 15, hopper_y + hopper_radius + flame_length)
            ]
            pygame.draw.polygon(self.screen, self.thrust_color, flame_points)
    
    def draw_target_and_bands(self, target_height, hysteresis_band):
        """
        Draw the target height line and hysteresis bands.
        
        Args:
            target_height (float): Target height in meters
            hysteresis_band (float): Width of the hysteresis band in meters
        """
        # Target line
        target_y = self.world_to_screen_y(target_height)
        target_line_start = (0, target_y)
        target_line_end = (self.width - self.control_panel_width, target_y)
        
        # Draw target line
        pygame.draw.line(self.screen, self.target_color, target_line_start, target_line_end, 2)
        
        # Draw text label
        target_text = self.font.render(f"Target: {target_height:.2f}m", True, self.target_color)
        self.screen.blit(target_text, (10, target_y - 20))
        
        if self.show_hysteresis:
            # Upper and lower hysteresis bounds
            half_band = hysteresis_band / 2
            upper_y = self.world_to_screen_y(target_height + half_band)
            lower_y = self.world_to_screen_y(target_height - half_band)
            
            # Draw hysteresis bands
            upper_surface = pygame.Surface((self.width - self.control_panel_width, target_y - upper_y), pygame.SRCALPHA)
            upper_surface.fill(self.hysteresis_upper_color)
            self.screen.blit(upper_surface, (0, upper_y))
            
            lower_surface = pygame.Surface((self.width - self.control_panel_width, lower_y - target_y), pygame.SRCALPHA)
            lower_surface.fill(self.hysteresis_lower_color)
            self.screen.blit(lower_surface, (0, target_y))
            
            # Draw bound lines
            pygame.draw.line(self.screen, self.hysteresis_upper_color, 
                            (0, upper_y), (self.width - self.control_panel_width, upper_y), 1)
            pygame.draw.line(self.screen, self.hysteresis_lower_color, 
                            (0, lower_y), (self.width - self.control_panel_width, lower_y), 1)
    
    def draw_control_panel(self, physics_state, kalman_state, controller_data, display_fps):
        """
        Draw the control panel showing system state and controls.
        
        Args:
            physics_state (tuple): Current physics state (position, velocity, acceleration)
            kalman_state (tuple): Current Kalman filter state estimate (position, velocity, acceleration)
            controller_data (dict): Controller data including target height and hysteresis band
            display_fps (float): Current FPS to display
        """
        # Extract data
        true_pos, true_vel, true_acc = physics_state
        est_pos, est_vel, est_acc = kalman_state
        target_height = controller_data['target_height']
        hysteresis_band = controller_data['hysteresis_band']
        
        # Draw panel background
        pygame.draw.rect(self.screen, (220, 220, 220), self.control_panel_rect)
        pygame.draw.line(self.screen, (180, 180, 180), 
                        (self.width - self.control_panel_width, 0),
                        (self.width - self.control_panel_width, self.height), 2)
        
        # Title
        title = self.title_font.render("Pneumatic Hopper Control Panel", True, self.text_color)
        self.screen.blit(title, (self.width - self.control_panel_width + 20, 20))
        
        # System state
        y_offset = 70
        line_height = 25
        
        state_title = self.title_font.render("System State", True, self.text_color)
        self.screen.blit(state_title, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += 30
        
        # True state
        pos_text = self.font.render(f"True Position: {true_pos:.3f} m", True, self.text_color)
        vel_text = self.font.render(f"True Velocity: {true_vel:.3f} m/s", True, self.text_color)
        acc_text = self.font.render(f"True Acceleration: {true_acc:.3f} m/s²", True, self.text_color)
        
        self.screen.blit(pos_text, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += line_height
        self.screen.blit(vel_text, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += line_height
        self.screen.blit(acc_text, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += line_height + 10
        
        # Kalman state if enabled
        if self.show_kalman:
            kalman_title = self.title_font.render("Kalman Filter Estimate", True, self.text_color)
            self.screen.blit(kalman_title, (self.width - self.control_panel_width + 20, y_offset))
            y_offset += 30
            
            est_pos_text = self.font.render(f"Est. Position: {est_pos:.3f} m", True, self.text_color)
            est_vel_text = self.font.render(f"Est. Velocity: {est_vel:.3f} m/s", True, self.text_color)
            est_acc_text = self.font.render(f"Est. Acceleration: {est_acc:.3f} m/s²", True, self.text_color)
            
            self.screen.blit(est_pos_text, (self.width - self.control_panel_width + 20, y_offset))
            y_offset += line_height
            self.screen.blit(est_vel_text, (self.width - self.control_panel_width + 20, y_offset))
            y_offset += line_height
            self.screen.blit(est_acc_text, (self.width - self.control_panel_width + 20, y_offset))
            y_offset += line_height + 10
        
        # Controller parameters
        controller_title = self.title_font.render("Controller Parameters", True, self.text_color)
        self.screen.blit(controller_title, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += 30
        
        target_text = self.font.render(f"Target Height: {target_height:.2f} m", True, self.text_color)
        band_text = self.font.render(f"Hysteresis Band: ±{hysteresis_band/2:.2f} m", True, self.text_color)
        
        self.screen.blit(target_text, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += line_height
        self.screen.blit(band_text, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += line_height + 20
        
        # Controls help
        controls_title = self.title_font.render("Controls", True, self.text_color)
        self.screen.blit(controls_title, (self.width - self.control_panel_width + 20, y_offset))
        y_offset += 30
        
        controls = [
            "Space: Toggle Manual Thrust",
            "Up/Down: Adjust Target Height",
            "K: Toggle Kalman Filter Display",
            "H: Toggle Hysteresis Bands Display",
            "R: Reset Simulation"
        ]
        
        for control in controls:
            control_text = self.font.render(control, True, self.text_color)
            self.screen.blit(control_text, (self.width - self.control_panel_width + 20, y_offset))
            y_offset += line_height
        
        # FPS counter
        fps_text = self.font.render(f"FPS: {display_fps:.1f}", True, self.text_color)
        self.screen.blit(fps_text, (self.width - 100, self.height - 30))
    
    def draw_sensors(self, lidar_reading, mpu_reading, new_lidar, new_mpu):
        """
        Draw sensor readings and indicate when new readings are available.
        
        Args:
            lidar_reading (float): Current LiDAR reading (height)
            mpu_reading (float): Current MPU6050 reading (acceleration)
            new_lidar (bool): True if a new LiDAR reading was just received
            new_mpu (bool): True if a new MPU6050 reading was just received
        """
        # Draw LiDAR reading
        if lidar_reading is not None:
            lidar_y = self.world_to_screen_y(lidar_reading)
            lidar_color = (0, 200, 0) if new_lidar else (0, 100, 0)
            
            # Draw LiDAR line
            pygame.draw.line(self.screen, lidar_color, 
                            (0, lidar_y), (50, lidar_y), 2)
            
            # Draw LiDAR text
            lidar_text = self.font.render(f"LiDAR: {lidar_reading:.2f}m", True, lidar_color)
            self.screen.blit(lidar_text, (10, lidar_y - 20))
        
        # Draw MPU reading indication (just a text label in the corner)
        if mpu_reading is not None:
            mpu_color = (200, 0, 200) if new_mpu else (100, 0, 100)
            mpu_text = self.font.render(f"MPU: {mpu_reading:.2f} m/s²", True, mpu_color)
            self.screen.blit(mpu_text, (10, 10))
    
    def update(self, physics_state, kalman_state, controller_data, sensor_data):
        """
        Update the visualization.
        
        Args:
            physics_state (tuple): Current physics state (position, velocity, acceleration, thrust)
            kalman_state (tuple): Current Kalman filter state estimate (position, velocity, acceleration)
            controller_data (dict): Controller data including target height and hysteresis band
            sensor_data (tuple): Sensor data (lidar_reading, mpu_reading, new_lidar, new_mpu)
        
        Returns:
            list: List of events that occurred
        """
        # Extract data
        true_pos, true_vel, true_acc, thrust_level = physics_state
        lidar_reading, mpu_reading, new_lidar, new_mpu = sensor_data
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        display_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Draw everything
        self.draw_background()
        self.draw_target_and_bands(controller_data['target_height'], controller_data['hysteresis_band'])
        self.draw_hopper(true_pos, true_vel, thrust_level)
        self.draw_sensors(lidar_reading, mpu_reading, new_lidar, new_mpu)
        self.draw_control_panel(
            (true_pos, true_vel, true_acc),
            kalman_state,
            controller_data,
            display_fps
        )
        
        # Update the display
        pygame.display.flip()
        
        # Handle events
        events = pygame.event.get()
        
        # Limit frame rate
        self.clock.tick(60)
        
        return events
    
    def handle_events(self, events):
        """
        Process pygame events and return actions.
        
        Args:
            events (list): List of pygame events
            
        Returns:
            dict: Dictionary of actions to take
        """
        actions = {
            'quit': False,
            'reset': False,
            'toggle_thrust': False,
            'adjust_target': 0.0,
            'toggle_kalman': False,
            'toggle_hysteresis': False
        }
        
        for event in events:
            if event.type == pygame.QUIT:
                actions['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions['quit'] = True
                elif event.key == pygame.K_r:
                    actions['reset'] = True
                elif event.key == pygame.K_SPACE:
                    actions['toggle_thrust'] = True
                elif event.key == pygame.K_UP:
                    actions['adjust_target'] = 0.1
                elif event.key == pygame.K_DOWN:
                    actions['adjust_target'] = -0.1
                elif event.key == pygame.K_k:
                    self.show_kalman = not self.show_kalman
                    actions['toggle_kalman'] = True
                elif event.key == pygame.K_h:
                    self.show_hysteresis = not self.show_hysteresis
                    actions['toggle_hysteresis'] = True
        
        return actions
    
    def close(self):
        """Clean up and close the visualizer."""
        pygame.quit()


# Test code
if __name__ == "__main__":
    import time
    
    # Create visualizer
    viz = Visualizer()
    
    # Test values
    position = 2.0
    velocity = 0.0
    acceleration = -9.81
    thrust = 0.0
    
    est_position = 2.1
    est_velocity = 0.1
    est_acceleration = -9.7
    
    target_height = 3.0
    hysteresis_band = 0.2
    
    lidar_reading = 2.05
    mpu_reading = -9.75
    
    # Simple test loop
    running = True
    manual_thrust = False
    
    while running:
        # Update physics (simple simulation for testing)
        if manual_thrust:
            thrust = 0.8
            acceleration = thrust * 20.0 - 9.81
        else:
            thrust = 0.0
            acceleration = -9.81
        
        velocity += acceleration * 0.01
        position += velocity * 0.01
        
        # Bounce on ground
        if position <= 0:
            position = 0
            velocity = abs(velocity) * 0.5
        
        # Update visualizer
        events = viz.update(
            (position, velocity, acceleration, thrust),
            (est_position, est_velocity, est_acceleration),
            {'target_height': target_height, 'hysteresis_band': hysteresis_band},
            (lidar_reading, mpu_reading, False, False)
        )
        
        # Update estimated values with some lag
        est_position = 0.9 * est_position + 0.1 * position
        est_velocity = 0.9 * est_velocity + 0.1 * velocity
        est_acceleration = 0.9 * est_acceleration + 0.1 * acceleration
        
        # Update sensor readings occasionally
        if np.random.random() < 0.05:
            lidar_reading = position + np.random.normal(0, 0.05)
            lidar_new = True
        else:
            lidar_new = False
            
        if np.random.random() < 0.1:
            mpu_reading = acceleration + np.random.normal(0, 0.1)
            mpu_new = True
        else:
            mpu_new = False
        
        # Handle events
        actions = viz.handle_events(events)
        
        if actions['quit']:
            running = False
        
        if actions['reset']:
            position = 2.0
            velocity = 0.0
            acceleration = -9.81
            thrust = 0.0
            est_position = 2.1
            est_velocity = 0.1
            est_acceleration = -9.7
            
        if actions['toggle_thrust']:
            manual_thrust = not manual_thrust
            
        if actions['adjust_target'] != 0.0:
            target_height += actions['adjust_target']
            target_height = max(0.5, target_height)  # Keep target above ground
    
    # Clean up
    viz.close()
    print("Visualization test complete.")
