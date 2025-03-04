"""
Menu system for the pneumatic hopper simulation.
Provides a pop-up menu interface for adjusting parameters.
"""

import pygame
import pygame_gui
import numpy as np

class SettingsMenu:
    """
    Pop-up menu for adjusting simulation parameters.
    """
    
    def __init__(self, width, height, physics_engine, controller, visualizer):
        """
        Initialize the settings menu.
        
        Args:
            width (int): Menu width in pixels
            height (int): Menu height in pixels
            physics_engine: Reference to the physics engine instance
            controller: Reference to the controller instance
            visualizer: Reference to the visualizer instance
        """
        self.width = width
        self.height = height
        self.physics = physics_engine
        self.controller = controller
        self.visualizer = visualizer
        
        # Store initial values for reset functionality
        self.initial_physics_values = {
            'mass': self.physics.mass,
            'max_thrust': self.physics.max_thrust,
            'delay_time': self.physics.delay_time,
            'air_resistance': self.physics.air_resistance
        }
        
        # Current control method
        self.control_methods = ["Hysteresis", "PID", "Bang-Bang"]
        if hasattr(self.controller, 'hysteresis_band'):
            self.current_control_method = "Hysteresis"
        elif hasattr(self.controller, 'kp'):
            self.current_control_method = "PID"
        else:
            self.current_control_method = "Bang-Bang"
        
        # Initialize pygame_gui manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Create panel background
        self.panel_rect = pygame.Rect((pygame.display.get_surface().get_width() - width) // 2, 
                                      (pygame.display.get_surface().get_height() - height) // 2, 
                                      width, height)
        
        # Create UI elements
        self.create_ui_elements()
        
        # Apply initial values from physics and controller
        self.update_ui_from_parameters()
        
        # Flags
        self.is_visible = False
        self.changes_made = False
        self.apply_changes = False
        self.needs_reset = False
    
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
            text="Simulation Settings",
            manager=self.ui_manager
        )
        y_pos += 40
        
        # Tabs for different setting categories
        tab_height = 30
        tab_width = (self.width - 2*panel_padding) // 3
        
        self.physics_tab_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_padding, y_pos), (tab_width, tab_height)),
            text="Physics",
            manager=self.ui_manager
        )
        
        self.control_tab_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_padding + tab_width, y_pos), (tab_width, tab_height)),
            text="Control",
            manager=self.ui_manager
        )
        
        self.visual_tab_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_padding + 2*tab_width, y_pos), (tab_width, tab_height)),
            text="Visual",
            manager=self.ui_manager
        )
        
        y_pos += tab_height + 10
        
        # Content container area
        self.content_height = self.height - y_pos - panel_padding - 50  # Space for buttons at bottom
        
        # --- Physics Parameters ---
        self.physics_container = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((panel_padding, y_pos), 
                                     (self.width - 2*panel_padding, self.content_height)),
            manager=self.ui_manager
        )
        
        phy_y_pos = 10
        
        # Mass Slider
        self.mass_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, phy_y_pos), (label_width, 25)),
            text="Mass (kg):",
            manager=self.ui_manager,
            container=self.physics_container
        )
        self.mass_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10 + label_width, phy_y_pos), (slider_width - 20, 25)),
            start_value=self.physics.mass,
            value_range=(0.1, 5.0),
            manager=self.ui_manager,
            container=self.physics_container
        )
        phy_y_pos += element_spacing
        
        # Max Thrust Slider
        self.thrust_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, phy_y_pos), (label_width, 25)),
            text="Max Thrust (N):",
            manager=self.ui_manager,
            container=self.physics_container
        )
        self.thrust_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10 + label_width, phy_y_pos), (slider_width - 20, 25)),
            start_value=self.physics.max_thrust,
            value_range=(5.0, 50.0),
            manager=self.ui_manager,
            container=self.physics_container
        )
        phy_y_pos += element_spacing
        
        # Pneumatic Delay Slider
        self.delay_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, phy_y_pos), (label_width, 25)),
            text="Delay (s):",
            manager=self.ui_manager,
            container=self.physics_container
        )
        self.delay_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10 + label_width, phy_y_pos), (slider_width - 20, 25)),
            start_value=self.physics.delay_time,
            value_range=(0.0, 0.5),
            manager=self.ui_manager,
            container=self.physics_container
        )
        phy_y_pos += element_spacing
        
        # Air Resistance Slider
        self.air_res_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, phy_y_pos), (label_width, 25)),
            text="Air Resistance:",
            manager=self.ui_manager,
            container=self.physics_container
        )
        self.air_res_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10 + label_width, phy_y_pos), (slider_width - 20, 25)),
            start_value=self.physics.air_resistance,
            value_range=(0.0, 0.5),
            manager=self.ui_manager,
            container=self.physics_container
        )
        
        # --- Controller Parameters ---
        self.control_container = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((panel_padding, y_pos), 
                                     (self.width - 2*panel_padding, self.content_height)),
            manager=self.ui_manager
        )
        
        ctrl_y_pos = 10
        
        # Control Method Dropdown
        self.method_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, ctrl_y_pos), (label_width, 25)),
            text="Control Method:",
            manager=self.ui_manager,
            container=self.control_container
        )
        self.method_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=self.control_methods,
            starting_option=self.current_control_method,
            relative_rect=pygame.Rect((10 + label_width, ctrl_y_pos), (slider_width - 20, 25)),
            manager=self.ui_manager,
            container=self.control_container
        )
        ctrl_y_pos += element_spacing + 10
        
        # --- Hysteresis Parameters ---
        self.hysteresis_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((10, ctrl_y_pos), 
                                     (self.control_container.rect.width - 20, 70)),
            manager=self.ui_manager,
            container=self.control_container
        )
        
        hys_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 5), (self.hysteresis_panel.rect.width - 10, 20)),
            text="Hysteresis Controller Settings",
            manager=self.ui_manager,
            container=self.hysteresis_panel
        )
        
        # Hysteresis Band Width Slider
        band_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 30), (label_width, 25)),
            text="Band Width:",
            manager=self.ui_manager,
            container=self.hysteresis_panel
        )
        self.band_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((5 + label_width, 30), (self.hysteresis_panel.rect.width - 10 - label_width, 25)),
            start_value=getattr(self.controller, 'hysteresis_band', 0.3),
            value_range=(0.05, 1.0),
            manager=self.ui_manager,
            container=self.hysteresis_panel
        )
        
        # --- PID Parameters ---
        self.pid_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((10, ctrl_y_pos), 
                                     (self.control_container.rect.width - 20, 140)),
            manager=self.ui_manager,
            container=self.control_container
        )
        
        pid_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 5), (self.pid_panel.rect.width - 10, 20)),
            text="PID Controller Settings",
            manager=self.ui_manager,
            container=self.pid_panel
        )
        
        # P Gain Slider
        p_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 30), (label_width, 25)),
            text="P Gain:",
            manager=self.ui_manager,
            container=self.pid_panel
        )
        self.kp_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((5 + label_width, 30), (self.pid_panel.rect.width - 10 - label_width, 25)),
            start_value=getattr(self.controller, 'kp', 1.0),
            value_range=(0.0, 10.0),
            manager=self.ui_manager,
            container=self.pid_panel
        )
        
        # I Gain Slider
        i_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 65), (label_width, 25)),
            text="I Gain:",
            manager=self.ui_manager,
            container=self.pid_panel
        )
        self.ki_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((5 + label_width, 65), (self.pid_panel.rect.width - 10 - label_width, 25)),
            start_value=getattr(self.controller, 'ki', 0.1),
            value_range=(0.0, 2.0),
            manager=self.ui_manager,
            container=self.pid_panel
        )
        
        # D Gain Slider
        d_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 100), (label_width, 25)),
            text="D Gain:",
            manager=self.ui_manager,
            container=self.pid_panel
        )
        self.kd_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((5 + label_width, 100), (self.pid_panel.rect.width - 10 - label_width, 25)),
            start_value=getattr(self.controller, 'kd', 0.5),
            value_range=(0.0, 5.0),
            manager=self.ui_manager,
            container=self.pid_panel
        )
        
        # --- Bang-Bang Parameters ---
        self.bang_bang_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((10, ctrl_y_pos), 
                                     (self.control_container.rect.width - 20, 70)),
            manager=self.ui_manager,
            container=self.control_container
        )
        
        bb_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 5), (self.bang_bang_panel.rect.width - 10, 20)),
            text="Bang-Bang Controller Settings",
            manager=self.ui_manager,
            container=self.bang_bang_panel
        )
        
        # Threshold Slider
        threshold_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((5, 30), (label_width, 25)),
            text="Threshold:",
            manager=self.ui_manager,
            container=self.bang_bang_panel
        )
        self.threshold_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((5 + label_width, 30), (self.bang_bang_panel.rect.width - 10 - label_width, 25)),
            start_value=getattr(self.controller, 'threshold', 0.1),
            value_range=(0.01, 0.5),
            manager=self.ui_manager,
            container=self.bang_bang_panel
        )
        
        # --- Visualization Parameters ---
        self.visual_container = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((panel_padding, y_pos), 
                                     (self.width - 2*panel_padding, self.content_height)),
            manager=self.ui_manager
        )
        
        vis_y_pos = 10
        
        # Visualization Options
        vis_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, vis_y_pos), (self.visual_container.rect.width - 20, 25)),
            text="Display Options",
            manager=self.ui_manager,
            container=self.visual_container
        )
        vis_y_pos += 35
        
        # LiDAR Visualization Checkbox
        self.lidar_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, vis_y_pos), (25, 25)),
            text="✓" if self.visualizer.show_lidar_detail else "",
            manager=self.ui_manager,
            container=self.visual_container,
            tool_tip_text="Toggle LiDAR Beam Visualization"
        )
        self.lidar_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((40, vis_y_pos), (self.visual_container.rect.width - 50, 25)),
            text="Enhanced LiDAR Visualization",
            manager=self.ui_manager,
            container=self.visual_container
        )
        vis_y_pos += element_spacing
        
        # Kalman Filter Visualization Checkbox
        self.kalman_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, vis_y_pos), (25, 25)),
            text="✓" if self.visualizer.show_kalman else "",
            manager=self.ui_manager,
            container=self.visual_container,
            tool_tip_text="Toggle Kalman Filter Display"
        )
        self.kalman_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((40, vis_y_pos), (self.visual_container.rect.width - 50, 25)),
            text="Show Kalman Filter Estimates",
            manager=self.ui_manager,
            container=self.visual_container
        )
        vis_y_pos += element_spacing
        
        # Hysteresis Band Visualization Checkbox
        self.hysteresis_vis_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, vis_y_pos), (25, 25)),
            text="✓" if self.visualizer.show_hysteresis else "",
            manager=self.ui_manager,
            container=self.visual_container,
            tool_tip_text="Toggle Hysteresis Bands Display"
        )
        self.hysteresis_vis_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((40, vis_y_pos), (self.visual_container.rect.width - 50, 25)),
            text="Show Hysteresis Bands",
            manager=self.ui_manager,
            container=self.visual_container
        )
        
        # Action Buttons (at the bottom)
        y_pos = self.height - panel_padding - 40
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
            text="Reset Settings",
            manager=self.ui_manager
        )
        
        # Set initial active tab
        self.active_tab = "Physics"
        self.update_tab_visibility()
    
    def update_tab_visibility(self):
        """Update which tab content is visible based on active tab."""
        self.physics_container.hide()
        self.control_container.hide()
        self.visual_container.hide()
        
        if self.active_tab == "Physics":
            self.physics_container.show()
            self.physics_tab_button.select()
            self.control_tab_button.unselect()
            self.visual_tab_button.unselect()
        elif self.active_tab == "Control":
            self.control_container.show()
            self.physics_tab_button.unselect()
            self.control_tab_button.select()
            self.visual_tab_button.unselect()
            # Update controller panels based on selected method
            self.update_controller_ui_visibility()
        elif self.active_tab == "Visual":
            self.visual_container.show()
            self.physics_tab_button.unselect()
            self.control_tab_button.unselect()
            self.visual_tab_button.select()
    
    def update_controller_ui_visibility(self):
        """Update the visibility of controller-specific panels."""
        self.hysteresis_panel.hide()
        self.pid_panel.hide()
        self.bang_bang_panel.hide()
        
        if self.current_control_method == "Hysteresis":
            self.hysteresis_panel.show()
        elif self.current_control_method == "PID":
            self.pid_panel.show()
        elif self.current_control_method == "Bang-Bang":
            self.bang_bang_panel.show()
    
    def update_ui_from_parameters(self):
        """Update UI elements to reflect the current parameter values."""
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
        
        # Visual settings
        self.update_checkboxes()
    
    def update_checkboxes(self):
        """Update checkbox states based on current settings."""
        # LiDAR checkbox
        self.lidar_checkbox.set_text("✓" if self.visualizer.show_lidar_detail else "")
        
        # Kalman Filter checkbox
        self.kalman_checkbox.set_text("✓" if self.visualizer.show_kalman else "")
        
        # Hysteresis Band checkbox
        self.hysteresis_vis_checkbox.set_text("✓" if self.visualizer.show_hysteresis else "")
    
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
            'reset_params': False,
            'apply_changes': False,
            'lidar_detail_changed': False,
            'kalman_display_changed': False,
            'hysteresis_display_changed': False
        }
        
        # Skip event handling if not visible
        if not self.is_visible:
            return changes
        
        # Process the event through pygame_gui
        ui_processed = self.ui_manager.process_events(event)
        
        # Handle UI interactions
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.physics_tab_button:
                self.active_tab = "Physics"
                self.update_tab_visibility()
            elif event.ui_element == self.control_tab_button:
                self.active_tab = "Control"
                self.update_tab_visibility()
            elif event.ui_element == self.visual_tab_button:
                self.active_tab = "Visual"
                self.update_tab_visibility()
            elif event.ui_element == self.apply_button:
                changes['apply_changes'] = True
                self.apply_parameter_changes()
            elif event.ui_element == self.reset_params_button:
                changes['reset_params'] = True
                self.reset_parameters()
            elif event.ui_element == self.lidar_checkbox:
                self.visualizer.show_lidar_detail = not self.visualizer.show_lidar_detail
                changes['lidar_detail_changed'] = True
                self.update_checkboxes()
            elif event.ui_element == self.kalman_checkbox:
                self.visualizer.show_kalman = not self.visualizer.show_kalman
                changes['kalman_display_changed'] = True
                self.update_checkboxes()
            elif event.ui_element == self.hysteresis_vis_checkbox:
                self.visualizer.show_hysteresis = not self.visualizer.show_hysteresis
                changes['hysteresis_display_changed'] = True
                self.update_checkboxes()
        
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.method_dropdown:
                self.current_control_method = event.text
                self.update_controller_ui_visibility()
                changes['control_method_changed'] = True
                self.changes_made = True
        
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            # Track that changes have been made
            self.changes_made = True
        
        # Handle click outside menu area to close
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # If click is outside the panel rect, hide the menu
                if not self.panel_rect.collidepoint(event.pos) and ui_processed == False:
                    self.is_visible = False
        
        return changes
    
    def apply_parameter_changes(self):
        """Apply the current parameter settings to the simulation."""
        if not self.changes_made:
            return
            
        # Apply physics parameters
        self.physics.mass = self.mass_slider.get_current_value()
        self.physics.max_thrust = self.thrust_slider.get_current_value()
        
        # Update delay time (this requires special handling due to the delay steps)
        new_delay = self.delay_slider.get_current_value()
        self.physics.delay_time = new_delay
        self.physics.delay_steps = int(new_delay / self.physics.dt)
        self.physics.control_history = [0.0] * self.physics.delay_steps
        
        self.physics.air_resistance = self.air_res_slider.get_current_value()
        
        # Controller parameters are set during method changes
        
        # Set flag to indicate changes have been applied
        self.apply_changes = True
        self.changes_made = False
        self.needs_reset = True
    
    def update(self, time_delta):
        """
        Update the UI manager and any dynamic elements.
        
        Args:
            time_delta (float): Time passed since last update
        """
        if self.is_visible:
            self.ui_manager.update(time_delta)
    
    def draw(self, surface):
        """
        Draw the menu to the given surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        if self.is_visible:
            # Draw a semi-transparent dark overlay behind the panel
            overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # RGBA, last value is alpha (transparency)
            surface.blit(overlay, (0, 0))
            
            # Draw panel background
            pygame.draw.rect(surface, (240, 240, 240), self.panel_rect)
            pygame.draw.rect(surface, (100, 100, 100), self.panel_rect, 2)  # Border
            
            # Draw UI elements
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
        
        # Reset visualization settings
        self.visualizer.show_lidar_detail = True
        self.visualizer.show_kalman = True
        self.visualizer.show_hysteresis = True
        
        # Reset UI elements
        self.update_ui_from_parameters()
        
        # Set flag to indicate simulation needs reset
        self.needs_reset = True
    
    def set_visible(self, visible):
        """Set whether the menu is visible."""
        self.is_visible = visible
        
        # If becoming visible, update UI from current parameters
        if visible:
            self.update_ui_from_parameters()
            
    def needs_simulation_reset(self):
        """Check if the simulation needs to be reset due to parameter changes."""
        if self.needs_reset:
            self.needs_reset = False
            return True
        return False
