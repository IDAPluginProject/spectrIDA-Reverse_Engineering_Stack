"""
Goal Representations

Goals define WHAT the agent is trying to achieve.
The planner uses goals to evaluate imagined futures.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class Goal(ABC):
    """Abstract goal interface."""

    @abstractmethod
    def satisfaction_score(self, state: torch.Tensor) -> torch.Tensor:
        """
        How well does this state satisfy the goal?
        
        Args:
            state: [batch, state_dim] or [state_dim]
            
        Returns:
            score: scalar or [batch] — higher is better
        """
        pass

    @abstractmethod
    def satisfaction_score_np(self, state: np.ndarray) -> float:
        """NumPy version for non-torch contexts."""
        pass


class PositionGoal(Goal):
    """
    Goal: reach a target position.
    
    The state is expected to have the first 2 dimensions
    as (x, y) coordinates.
    """

    def __init__(self, target_position: torch.Tensor, tolerance: float = 0.5):
        """
        Args:
            target_position: [2] or [batch, 2] target x, y
            tolerance: how close is close enough
        """
        self.target = target_position
        self.tolerance = tolerance
    
    def satisfaction_score(self, state: torch.Tensor) -> torch.Tensor:
        """Score based on distance to target (closer = higher)."""
        if state.dim() == 1:
            pos = state[:2]
        else:
            pos = state[:, :2]
        
        target = self.target
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        distance = torch.norm(pos - target, dim=-1)
        
        # Gaussian-like scoring: 1.0 at distance 0, falls off with tolerance
        score = torch.exp(-0.5 * (distance / self.tolerance) ** 2)
        
        return score
    
    def satisfaction_score_np(self, state: np.ndarray) -> float:
        """NumPy version."""
        pos = state[:2]
        distance = np.linalg.norm(pos - self.target.cpu().numpy())
        return float(np.exp(-0.5 * (distance / self.tolerance) ** 2))


class CompositeGoal(Goal):
    """
    Combine multiple sub-goals with weights.
    
    Example: "reach the door AND pick up the key"
    """

    def __init__(self, goals: list[Goal], weights: list[float] = None):
        self.goals = goals
        if weights is None:
            weights = [1.0 / len(goals)] * len(goals)
        self.weights = weights
    
    def satisfaction_score(self, state: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=state.device)
        for goal, weight in zip(self.goals, self.weights):
            total = total + weight * goal.satisfaction_score(state)
        return total
    
    def satisfaction_score_np(self, state: np.ndarray) -> float:
        total = 0.0
        for goal, weight in zip(self.goals, self.weights):
            total += weight * goal.satisfaction_score_np(state)
        return total
