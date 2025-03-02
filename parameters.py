"""
Parameter management and GUI controls for the pneumatic hopper simulation.
Provides a UI for adjusting simulation parameters in real-time.
"""

import pygame
import pygame_gui
import numpy as np

class ParameterPanel:
    """
    GUI panel for adjusting simulation parameters in real-time.
    """
    
    def __init__(self, width, height, physics_engine, controller):
        """
        Initialize the parameter panel.
        
        Args:
            width (int): Panel width in pixels
            height (int): Panel height in pixels
            physics_engine: Reference to the physics engine instance
            controller: Reference to the controller instance
        """
        self.width = width
        self.height = height
        self.physics = physics_engine
        self.controller = controller
        
        # Current control method
        self.control_methods = ["Hysteresis", "PID", "Bang-Bang"]
        self.current_control_method = "Hysteresis"
        
        # Initialize pygame_gui manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Create panel background
        self.panel_rect = pygame.Rect(0, 0, width, height)
        
        # Create UI elements
        self.create_ui_elements()
        
        # Apply initial values from physics and controller
        self.update_ui_from_parameters()
    
    def create_ui_elements(self):
        """Create all UI elements for parameter adjustment."""
        # Spacing and positioning parameters
        panel_padding = 20
        element_spacing = 35
        label_width = 150
        slider_width = self.width - 2*panel_padding - label_width
        
        # Starting Y position
        y_pos = panel_padding
        
        # Title
        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((panel_padding, y_pos), (self.width - 2*panel_padding, 30)),
            text="Parameter Adjustment Panel",
            manager=self.ui_manager
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
            start_value=self.controller.hysteresis_band,
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
            start_value=1.0,  # Default value
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
            start_value=0.1,  # Default value
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
            start_value=0.5,  # Default value
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
            start_value=0.1,  # Default value
            value_range=(0.01, 0.5),
            manager=self.ui_manager
        )
        y_pos += element_spacing + 20
        
        # Reset Button
        button_width = 120
        self.reset_params_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(((self.width - button_width) // 2, y_pos), (button_width, 30)),
            text="Reset Parameters",
            manager=self.ui_manager
        )
    
    def update_ui_from_parameters(self):
        """Update UI elements to reflect the current parameter values."""
        # Physics parameters
        self.mass_slider.set_current_value(self.physics.mass)
        self.thrust_slider.set_current_value(self.physics.max_thrust)
        self.delay_slider.set_current_value(self.physics.delay_time)
        self.air_res_slider.set_current_value(self.physics.air_resistance)
        
        # Controller parameters
        self.band_slider.set_current_value(self.controller.hysteresis_band)
        
        # Set visibility of controller-specific parameters based on current control method
        self.update_controller_ui_visibility()
    
    def update_controller_ui_visibility(self):
        """Update the visibility of controller-specific UI elements."""
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
    
    def handle_event(self, event):
        """
        Handle pygame and UI events.
        
        Args:
            event: Pygame event object
            
        Returns:
            dict: Dictionary containing parameter changes
        """
        changes = {
            'physics_changed': False,
            'controller_changed': False,
            'control_method_changed': False,
            'reset_params': False
        }
        
        # Process the event through pygame_gui
        self.ui_manager.process_events(event)
        
        # Handle UI interactions
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            # Physics parameters
            if event.ui_element == self.mass_slider:
                self.physics.mass = self.mass_slider.get_current_value()
                changes['physics_changed'] = True
            elif event.ui_element == self.thrust_slider:
                self.physics.max_thrust = self.thrust_slider.get_current_value()
                changes['physics_changed'] = True
            elif event.ui_element == self.delay_slider:
                new_delay = self.delay_slider.get_current_value()
                self.physics.delay_time = new_delay
                self.physics.delay_steps = int(new_delay / self.physics.dt)
                self.physics.control_history = [0.0] * self.physics.delay_steps
                changes['physics_changed'] = True
            elif event.ui_element == self.air_res_slider:
                self.physics.air_resistance = self.air_res_slider.get_current_value()
                changes['physics_changed'] = True
            
            # Controller parameters
            elif event.ui_element == self.band_slider:
                self.controller.hysteresis_band = self.band_slider.get_current_value()
                changes['controller_changed'] = True
            elif event.ui_element == self.kp_slider or event.ui_element == self.ki_slider or event.ui_element == self.kd_slider:
                changes['controller_changed'] = True
            elif event.ui_element == self.threshold_slider:
                changes['controller_changed'] = True
        
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.method_dropdown:
                self.current_control_method = event.text
                self.update_controller_ui_visibility()
                changes['control_method_changed'] = True
        
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.reset_params_button:
                changes['reset_params'] = True
        
        return changes
    
    def update(self, time_delta):
        """
        Update the UI manager and any dynamic elements.
        
        Args:
            time_delta (float): Time passed since last update
        """
        self.ui_manager.update(time_delta)
    
    def draw(self, surface):
        """
        Draw the parameter panel to the given surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        pygame.draw.rect(surface, (240, 240, 240), self.panel_rect)
        self.ui_manager.draw_ui(surface)
    
    def get_controller_parameters(self):
        """
        Get the current controller parameters based on the selected control method.
        
        Returns:
            dict: Dictionary of controller parameters
        """
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
        
        return params
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        # Reset physics parameters
        self.physics.mass = 1.0
        self.physics.max_thrust = 20.0
        self.physics.delay_time = 0.2
        self.physics.air_resistance = 0.1
        self.physics.delay_steps = int(self.physics.delay_time / self.physics.dt)
        self.physics.control_history = [0.0] * self.physics.delay_steps
        
        # Reset controller parameters
        self.controller.hysteresis_band = 0.3
        
        # Reset UI elements
        self.update_ui_from_parameters()
