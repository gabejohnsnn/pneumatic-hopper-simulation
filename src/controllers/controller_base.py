#!/usr/bin/env python3
"""
Controller Base - Abstract base class for all controllers

This module defines the abstract base class that all controllers should inherit from
to ensure a consistent interface across different control methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Any


class ControllerBase(ABC):
    """
    Abstract base class that defines the interface for all controllers.
    
    All controller implementations should inherit from this class and implement
    the required methods to ensure a consistent interface.
    """
    
    def __init__(self, target_height: float = 3.0):
        """
        Initialize the controller with a target height.
        
        Args:
            target_height (float): Initial target height in meters.
        """
        self._target_height = target_height
    
    @abstractmethod
    def compute_action(self, estimated_state: Union[Dict[str, float], Tuple[float, float, float]], 
                      target_height: float = None) -> float:
        """
        Compute control action based on estimated state and target height.
        
        Args:
            estimated_state: The estimated state from the Kalman filter.
                             Can be a dictionary {'position': pos, 'velocity': vel, 'acceleration': acc}
                             or a tuple (position, velocity, acceleration).
            target_height: Target height to reach. If None, uses the controller's stored target height.
            
        Returns:
            float: Control action between 0 and 1 (normalized thrust).
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the controller to its initial state.
        
        Should clear any internal state variables or histories.
        """
        pass
    
    def set_target_height(self, height: float) -> None:
        """
        Set a new target height for the controller.
        
        Args:
            height (float): New target height in meters.
        """
        self._target_height = height
    
    def get_target_height(self) -> float:
        """
        Get the current target height.
        
        Returns:
            float: Current target height in meters.
        """
        return self._target_height
    
    def adjust_target_height(self, delta: float) -> None:
        """
        Adjust the target height by a delta.
        
        Args:
            delta (float): Amount to change the target height by.
        """
        self._target_height += delta
        # Ensure target height stays within reasonable bounds
        self._target_height = max(0.1, min(10.0, self._target_height))
