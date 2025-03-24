import pygame
import pygame_gui
import numpy as np

class ParameterPanel:
    
    def __init__(self, width, height, physics_engine, controller, lidar_detail_enabled=True):
        self.width = width
        self.height = height
        self.physics = physics_engine
        self.controller = controller
        self.lidar_detail_enabled = lidar_detail_enabled
        
        # Store initial values for reset functionality
        self.initial_physics_values = {
            'mass': self.physics.mass,
            'max_thrust': self.physics.max_thrust,
            'delay_time': self.physics.delay_time,
            'air_resistance': self.physics.air_resistance
        }
        
        # Current control method
        self.control_methods = ["Hysteresis", "PID", "Bang-Bang", "DDPG"]
        
        # Determine current control method from controller instance
        if hasattr(self.controller, 'hysteresis_band'):
            self.current_control_method = "Hysteresis"
        elif hasattr(self.controller, 'kp'):
            self.current_control_method = "PID"
        elif hasattr(self.controller, 'threshold'):
            self.current_control_method = "Bang-Bang"
        elif hasattr(self.controller, 'actor'):
            self.current_control_method = "DDPG"
        else:
            self.current_control_method = "Hysteresis"  # Default
        
        # Create a separate surface for the panel
        self.panel_surface = None
        
        # Initialize pygame_gui manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Create panel background
        self.panel_rect = pygame.Rect(0, 0, width, height)
        
        # Create UI elements
        self.create_ui_elements()
        
        # Apply initial values from physics and controller
        self.update_ui_from_parameters()
        
        # Flag to indicate if changes should be applied
        self.apply_changes = False
        self.is_visible = False
        self.initialized = False
    
    def create_ui_elements(self):
        # Spacing and positioning parameters
        panel_padding = 20
        element_spacing = 35
        label_width = 150
        slider_width = self.width - 2*panel_padding - label_width
        
        # Starting Y position
        y_pos = panel_padding
        
        # Title
        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (self.width - 2*panel_padding - 30, 30)),
            text="Parameter Adjustment Panel",
            manager=self.ui_manager
        )
        
        # Close button (X in top-right corner)
        self.close_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.width - panel_padding - 25, y_pos), (25, 25)),
            text="X",
            manager=self.ui_manager,
            tool_tip_text="Close Parameter Panel"
        )
        
        y_pos += 40
        
        # Physics Parameters Section
        self.physics_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (self.width - 2*panel_padding, 25)),
            text="Physics Parameters",
            manager=self.ui_manager
        )
        y_pos += 30
        
        # Mass Slider
        self.mass_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Mass (kg):",
            manager=self.ui_manager
        )
        self.mass_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=self.physics.mass,
            value_range=(0.1, 5.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Max Thrust Slider
        self.thrust_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Max Thrust (N):",
            manager=self.ui_manager
        )
        self.thrust_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=self.physics.max_thrust,
            value_range=(5.0, 50.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Pneumatic Delay Slider
        self.delay_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Delay (s):",
            manager=self.ui_manager
        )
        self.delay_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=self.physics.delay_time,
            value_range=(0.0, 0.5),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Air Resistance Slider
        self.air_res_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Air Resistance:",
            manager=self.ui_manager
        )
        self.air_res_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=self.physics.air_resistance,
            value_range=(0.0, 0.5),
            manager=self.ui_manager
        )
        y_pos += element_spacing + 10
        
        # Controller Parameters Section
        self.controller_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (self.width - 2*panel_padding, 25)),
            text="Controller Parameters",
            manager=self.ui_manager
        )
        y_pos += 30
        
        # Control Method Dropdown
        self.method_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Control Method:",
            manager=self.ui_manager
        )
        self.method_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=self.control_methods,
            starting_option=self.current_control_method,
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # --- Hysteresis Parameters ---
        # Hysteresis Band Width Slider
        self.band_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Hysteresis Band:",
            manager=self.ui_manager
        )
        self.band_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'hysteresis_band', 0.3),
            value_range=(0.05, 1.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # --- PID Parameters ---
        # Proportional Gain Slider
        self.kp_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="P Gain:",
            manager=self.ui_manager
        )
        self.kp_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'kp', 1.0),
            value_range=(0.0, 10.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Integral Gain Slider
        self.ki_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="I Gain:",
            manager=self.ui_manager
        )
        self.ki_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'ki', 0.1),
            value_range=(0.0, 2.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Derivative Gain Slider
        self.kd_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="D Gain:",
            manager=self.ui_manager
        )
        self.kd_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'kd', 0.5),
            value_range=(0.0, 5.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # --- Bang-Bang Parameters ---
        # Threshold Slider
        self.threshold_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Threshold:",
            manager=self.ui_manager
        )
        self.threshold_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'threshold', 0.1),
            value_range=(0.01, 0.5),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # --- DDPG Parameters ---
        # Learning Rate Slider
        self.lr_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Learning Rate:",
            manager=self.ui_manager
        )
        
        # Get default actor_lr or use a default value if not available
        default_lr = 1e-4
        if hasattr(self.controller, 'actor_optimizer') and hasattr(self.controller.actor_optimizer, 'param_groups'):
            if len(self.controller.actor_optimizer.param_groups) > 0:
                default_lr = self.controller.actor_optimizer.param_groups[0]['lr']
        
        self.lr_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=default_lr * 10000,  # Scale up for slider (1e-4 -> 1.0)
            value_range=(0.01, 10.0),  # 1e-6 to 1e-3
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # Noise Scale Slider
        self.noise_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (label_width, 25)),
            text="Exploration Noise:",
            manager=self.ui_manager
        )
        self.noise_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((panel_padding + label_width, y_pos), (slider_width, 25)),
            start_value=getattr(self.controller, 'noise_scale', 0.2),
            value_range=(0.0, 1.0),
            manager=self.ui_manager
        )
        y_pos += element_spacing
        
        # DDPG Training Mode Checkbox
        self.training_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_padding, y_pos), (25, 25)),
            text="",
            manager=self.ui_manager,
            tool_tip_text="Toggle DDPG Training Mode"
        )
        self.training_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding + 30, y_pos), (self.width - panel_padding - 30, 25)),
            text="DDPG Training Mode",
            manager=self.ui_manager
        )
        y_pos += element_spacing + 10
        
        # Visualization Parameters Section
        self.viz_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (self.width - 2*panel_padding, 25)),
            text="Visualization Options",
            manager=self.ui_manager
        )
        y_pos += 30
        
        # LiDAR Visualization Checkbox
        self.lidar_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_padding, y_pos), (25, 25)),
            text="",
            manager=self.ui_manager,
            tool_tip_text="Toggle LiDAR Visualization"
        )
        self.lidar_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding + 30, y_pos), (self.width - panel_padding - 30, 25)),
            text="Enhanced LiDAR Visualization",
            manager=self.ui_manager
        )
        y_pos += element_spacing + 20
        
        # Buttons
        button_width = 120
        button_spacing = 20
        total_button_width = 2 * button_width + button_spacing
        button_start_x = (self.width - total_button_width) // 2
        
        # Apply Button
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((button_start_x, y_pos), (button_width, 30)),
            text="Apply Changes",
            manager=self.ui_manager
        )
        
        # Reset Button
        self.reset_params_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((button_start_x + button_width + button_spacing, y_pos), (button_width, 30)),
            text="Reset Parameters",
            manager=self.ui_manager
        )
        
        # Update checkboxes
        self._update_checkbox_states()
    
    def _update_checkbox_states(self):
        """Update checkbox states based on current settings."""
        # Update LiDAR checkbox
        if self.lidar_checkbox is not None:
            if self.lidar_detail_enabled:
                self.lidar_checkbox.set_text("✓")
            else:
                self.lidar_checkbox.set_text("")
        
        # Update DDPG training mode checkbox
        if self.training_checkbox is not None:
            if hasattr(self.controller, 'training_mode') and self.controller.training_mode:
                self.training_checkbox.set_text("✓")
            else:
                self.training_checkbox.set_text("")
    
    def update_ui_from_parameters(self):
        # Physics parameters
        self.mass_slider.set_current_value(self.physics.mass)
        self.thrust_slider.set_current_value(self.physics.max_thrust)
        self.delay_slider.set_current_value(self.physics.delay_time)
        self.air_res_slider.set_current_value(self.physics.air_resistance)
        
        # Controller parameters
        if hasattr(self.controller, 'hysteresis_band'):
            self.band_slider.set_current_value(self.controller.hysteresis_band)
        
        if hasattr(self.controller, 'kp'):
            self.kp_slider.set_current_value(self.controller.kp)
            self.ki_slider.set_current_value(self.controller.ki)
            self.kd_slider.set_current_value(self.controller.kd)
        
        if hasattr(self.controller, 'threshold'):
            self.threshold_slider.set_current_value(self.controller.threshold)
        
        # DDPG parameters
        if hasattr(self.controller, 'actor_optimizer') and hasattr(self.controller.actor_optimizer, 'param_groups'):
            if len(self.controller.actor_optimizer.param_groups) > 0:
                self.lr_slider.set_current_value(self.controller.actor_optimizer.param_groups[0]['lr'] * 10000)
        
        if hasattr(self.controller, 'noise_scale'):
            self.noise_slider.set_current_value(self.controller.noise_scale)
        
        # Set visibility of controller-specific parameters based on current control method
        self.update_controller_ui_visibility()
        
        # Update checkboxes
        self._update_checkbox_states()
    
    def update_controller_ui_visibility(self):
        # Hide all controller-specific parameters first
        self.band_label.hide()
        self.band_slider.hide()
        self.kp_label.hide()
        self.kp_slider.hide()
        self.ki_label.hide()
        self.ki_slider.hide()
        self.kd_label.hide()
        self.kd_slider.hide()
        self.threshold_label.hide()
        self.threshold_slider.hide()
        self.lr_label.hide()
        self.lr_slider.hide()
        self.noise_label.hide()
        self.noise_slider.hide()
        self.training_checkbox.hide()
        self.training_label.hide()
        
        # Show parameters specific to the current control method
        if self.current_control_method == "Hysteresis":
            self.band_label.show()
            self.band_slider.show()
        elif self.current_control_method == "PID":
            self.kp_label.show()
            self.kp_slider.show()
            self.ki_label.show()
            self.ki_slider.show()
            self.kd_label.show()
            self.kd_slider.show()
        elif self.current_control_method == "Bang-Bang":
            self.threshold_label.show()
            self.threshold_slider.show()
        elif self.current_control_method == "DDPG":
            self.lr_label.show()
            self.lr_slider.show()
            self.noise_label.show()
            self.noise_slider.show()
            self.training_checkbox.show()
            self.training_label.show()
    
    def handle_event(self, event):
        changes = {
            'physics_changed': False,
            'controller_changed': False,
            'control_method_changed': False,
            'reset_params': False,
            'apply_changes': False,
            'lidar_detail_changed': False
        }
        
        # Skip event handling if not visible
        if not self.is_visible:
            return changes
        
        # Process the event through pygame_gui
        self.ui_manager.process_events(event)
        
        # Handle UI interactions
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            # No need to immediately apply changes - will be done when Apply button is clicked
            pass
        
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.method_dropdown:
                self.current_control_method = event.text
                changes['control_method_changed'] = True
                self.update_controller_ui_visibility()
        
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.reset_params_button:
                changes['reset_params'] = True
                self.reset_parameters()
            elif event.ui_element == self.apply_button:
                changes['apply_changes'] = True
                self.apply_parameter_changes()
            elif event.ui_element == self.close_button:
                # Close the panel
                self.set_visible(False)
                changes['toggle_settings'] = True  # So main.py knows settings were toggled
            elif event.ui_element == self.lidar_checkbox:
                self.lidar_detail_enabled = not self.lidar_detail_enabled
                changes['lidar_detail_changed'] = True
                self._update_checkbox_states()
            elif event.ui_element == self.training_checkbox and self.current_control_method == "DDPG":
                if hasattr(self.controller, 'set_training_mode'):
                    new_training_mode = not getattr(self.controller, 'training_mode', True)
                    self.controller.set_training_mode(new_training_mode)
                    self._update_checkbox_states()
                    print(f"DDPG training mode set to: {new_training_mode}")
        
        return changes
    
    def apply_parameter_changes(self):
        # Apply physics parameters
        self.physics.mass = self.mass_slider.get_current_value()
        self.physics.max_thrust = self.thrust_slider.get_current_value()
        
        # Update delay time (this requires special handling due to the delay steps)
        new_delay = self.delay_slider.get_current_value()
        self.physics.delay_time = new_delay
        self.physics.delay_steps = int(new_delay / self.physics.dt)
        self.physics.control_history = [0.0] * self.physics.delay_steps
        
        self.physics.air_resistance = self.air_res_slider.get_current_value()
        
        # Apply controller parameters based on type
        if self.current_control_method == "Hysteresis" and hasattr(self.controller, 'hysteresis_band'):
            self.controller.hysteresis_band = self.band_slider.get_current_value()
        
        elif self.current_control_method == "PID" and hasattr(self.controller, 'kp'):
            self.controller.kp = self.kp_slider.get_current_value()
            self.controller.ki = self.ki_slider.get_current_value()
            self.controller.kd = self.kd_slider.get_current_value()
        
        elif self.current_control_method == "Bang-Bang" and hasattr(self.controller, 'threshold'):
            self.controller.threshold = self.threshold_slider.get_current_value()
        
        elif self.current_control_method == "DDPG":
            # Update learning rate
            if hasattr(self.controller, 'actor_optimizer') and hasattr(self.controller.actor_optimizer, 'param_groups'):
                new_lr = self.lr_slider.get_current_value() / 10000.0  # Scale down from slider (1.0 -> 1e-4)
                for param_group in self.controller.actor_optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                # Also update critic learning rate (typically 10x actor)
                if hasattr(self.controller, 'critic_optimizer'):
                    for param_group in self.controller.critic_optimizer.param_groups:
                        param_group['lr'] = new_lr * 10
                
                print(f"Updated DDPG learning rates: Actor={new_lr}, Critic={new_lr*10}")
            
            # Update exploration noise
            if hasattr(self.controller, 'noise_scale'):
                self.controller.noise_scale = self.noise_slider.get_current_value()
                print(f"Updated DDPG exploration noise scale: {self.controller.noise_scale}")
        
        # Set flag to indicate changes have been applied
        self.apply_changes = True
    
    def update(self, time_delta):
        # Only update UI manager when panel is visible
        if self.is_visible:
            self.ui_manager.update(time_delta)
    
    def draw(self, surface):
        """Draw the parameter panel on the main surface."""
        if self.is_visible:
            # Create semi-transparent overlay
            overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black
            surface.blit(overlay, (0, 0))
            
            # Draw panel background
            pygame.draw.rect(surface, (240, 240, 240), self.panel_rect)
            pygame.draw.rect(surface, (100, 100, 100), self.panel_rect, 2)  # Darker border
            
            # Draw title bar
            title_bar = pygame.Rect(self.panel_rect.left, self.panel_rect.top, self.panel_rect.width, 40)
            pygame.draw.rect(surface, (200, 200, 200), title_bar)
            
            # Draw UI elements
            self.ui_manager.draw_ui(surface)
    
    def get_controller_parameters(self):
        params = {
            'method': self.current_control_method
        }
        
        if self.current_control_method == "Hysteresis":
            params['band'] = self.band_slider.get_current_value()
        elif self.current_control_method == "PID":
            params['kp'] = self.kp_slider.get_current_value()
            params['ki'] = self.ki_slider.get_current_value()
            params['kd'] = self.kd_slider.get_current_value()
        elif self.current_control_method == "Bang-Bang":
            params['threshold'] = self.threshold_slider.get_current_value()
        elif self.current_control_method == "DDPG":
            params['learning_rate'] = self.lr_slider.get_current_value() / 10000.0
            params['noise_sigma'] = self.noise_slider.get_current_value()
            # Get training mode state
            params['train'] = True
            if hasattr(self.controller, 'training_mode'):
                params['train'] = self.controller.training_mode
        
        return params
    
    def reset_parameters(self):
        # Reset physics parameters
        self.physics.mass = self.initial_physics_values['mass']
        self.physics.max_thrust = self.initial_physics_values['max_thrust']
        self.physics.delay_time = self.initial_physics_values['delay_time']
        self.physics.air_resistance = self.initial_physics_values['air_resistance']
        self.physics.delay_steps = int(self.physics.delay_time / self.physics.dt)
        self.physics.control_history = [0.0] * self.physics.delay_steps
        
        # Reset to default controller parameters based on type
        if hasattr(self.controller, 'hysteresis_band'):
            self.controller.hysteresis_band = 0.3
        
        if hasattr(self.controller, 'kp'):
            self.controller.kp = 1.0
            self.controller.ki = 0.1
            self.controller.kd = 0.5
        
        if hasattr(self.controller, 'threshold'):
            self.controller.threshold = 0.1
        
        if self.current_control_method == "DDPG":
            # Reset DDPG specific parameters
            if hasattr(self.controller, 'noise'):
                self.controller.noise.reset()
                self.controller.noise_scale = 0.2
            
            if hasattr(self.controller, 'actor_optimizer') and hasattr(self.controller.actor_optimizer, 'param_groups'):
                for param_group in self.controller.actor_optimizer.param_groups:
                    param_group['lr'] = 1e-4
                
                if hasattr(self.controller, 'critic_optimizer'):
                    for param_group in self.controller.critic_optimizer.param_groups:
                        param_group['lr'] = 1e-3
        
        # Reset UI elements
        self.update_ui_from_parameters()
    
    def set_visible(self, visible):
        """Set whether the parameter panel is visible."""
        if self.is_visible != visible:
            print(f"Setting parameter panel visibility to: {visible}")
            self.is_visible = visible
            
            # On each visibility toggle, make sure the UI manager is properly setup
            # Get screen dimensions
            screen_info = pygame.display.Info()
            
            # Calculate center position
            panel_x = (screen_info.current_w - self.width) // 2
            panel_y = (screen_info.current_h - self.height) // 2
            self.panel_rect.topleft = (panel_x, panel_y)
            
            # Create a new UI manager with the correct position
            if visible:
                # Reset UI manager
                self.ui_manager = pygame_gui.UIManager((self.width, self.height))
                
                # Recreate all UI elements
                self.create_ui_elements()
                
                # Update with initial values
                self.update_ui_from_parameters()
                self.update_controller_ui_visibility()
                self._update_checkbox_states()
                
                print(f"Parameter panel initialized at: {panel_x}, {panel_y}")
    
    def get_lidar_detail_enabled(self):
        return self.lidar_detail_enabled
