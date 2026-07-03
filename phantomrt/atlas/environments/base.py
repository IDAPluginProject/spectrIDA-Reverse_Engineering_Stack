"""
Base Environment Interface

All environments inherit from this.
Provides a consistent API for the world model to interact with.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseEnvironment(ABC):
    """Abstract base class for all environments."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state. Returns observation."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take an action in the environment.
        
        Returns:
            observation: next observation
            reward: reward signal
            done: whether episode ended
            info: additional information
        """
        pass

    @abstractmethod
    def get_observation_dim(self) -> int:
        """Returns the dimensionality of observations."""
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        """Returns the dimensionality of actions."""
        pass

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render the environment. Returns RGB array or None."""
        pass

    def close(self):
        """Clean up resources."""
        pass
