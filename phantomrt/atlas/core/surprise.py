"""
Surprise Detection System

The KEY insight of the world model: only learn from SURPRISES.

If the model predicted reality correctly → no learning needed
If the model was WRONG → that's a surprise → UPDATE THE MODEL

This is how the brain works:
- You expect the floor to be there → you step on it → no surprise
- You expect the floor to be there → it's not → HUGE surprise → learning!

The surprise signal drives ALL learning in the model.
This is more efficient than backpropagating through everything —
we only allocate compute to things we got wrong.
"""

import torch
import torch.nn.functional as F
from collections import deque


class SurpriseDetector:
    """
    Detects when the world model's predictions don't match reality.
    
    Maintains an adaptive threshold:
    - Gets tighter as the model improves (higher standards)
    - Loosens when the environment changes (new things to learn)
    """

    def __init__(self, initial_threshold: float = 0.1, adaptation_rate: float = 0.01, history_size: int = 1000):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.history = deque(maxlen=history_size)
        
        # Track overall learning progress
        self.total_surprises = 0
        self.total_predictions = 0
        self.surprise_rate_history = deque(maxlen=100)
    
    def compute_surprise(self, real: torch.Tensor, predicted: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Compute how surprising the real observation is.
        
        Args:
            real: actual observation [batch, obs_dim]
            predicted: model's prediction [batch, obs_dim]
            
        Returns:
            (surprise_score, is_surprising): scalar loss + boolean
        """
        # Element-wise prediction error
        error = F.mse_loss(real, predicted, reduction="none").mean(dim=-1)  # [batch]
        
        # Overall surprise score (mean across batch)
        surprise_score = error.mean()
        
        # Track history
        self.history.append(surprise_score.item())
        self.total_predictions += real.shape[0]
        
        # Is this surprising enough to trigger learning?
        is_surprising = surprise_score.item() > self.threshold
        
        if is_surprising:
            self.total_surprises += real.shape[0]
        
        # Update adaptive threshold
        self._adapt_threshold()
        
        # Track surprise rate
        if self.total_predictions > 0:
            current_rate = self.total_surprises / self.total_predictions
            self.surprise_rate_history.append(current_rate)
        
        return surprise_score, is_surprising
    
    def _adapt_threshold(self):
        """
        Adapt the surprise threshold based on recent history.
        
        Logic:
        - If surprise rate is high (>50%): we're learning a lot, keep threshold
        - If surprise rate is low (<10%): model is good, tighten threshold
        - If surprise rate is medium: maintain current threshold
        """
        if len(self.history) < 50:
            return  # not enough data
        
        recent_surprises = list(self.history)[-50:]
        avg_surprise = sum(recent_surprises) / len(recent_surprises)
        
        # Adaptive: threshold follows average surprise but stays slightly below
        # This means ~50% of predictions will be "surprising"
        target = avg_surprise * 0.8
        
        # Smoothly adjust toward target
        self.threshold += self.adaptation_rate * (target - self.threshold)
        
        # Keep threshold in reasonable bounds
        self.threshold = max(0.001, min(self.threshold, 10.0))
    
    def compute_novelty(self, real: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty score — how DIFFERENT is this from what we've seen?
        
        High novelty = new situation = important to learn from
        Low novelty = familiar situation = can skip learning
        """
        error = F.mse_loss(real, predicted, reduction="none").mean(dim=-1)
        
        # Normalize by recent average surprise
        if len(self.history) > 0:
            avg = sum(self.history) / len(self.history)
            novelty = error / (avg + 1e-8)
        else:
            novelty = error
        
        return novelty
    
    def get_stats(self) -> dict:
        """Get current surprise statistics."""
        return {
            "threshold": self.threshold,
            "total_surprises": self.total_surprises,
            "total_predictions": self.total_predictions,
            "surprise_rate": (
                self.total_surprises / self.total_predictions
                if self.total_predictions > 0
                else 0.0
            ),
            "avg_surprise_recent": (
                sum(self.history) / len(self.history)
                if len(self.history) > 0
                else 0.0
            ),
        }
    
    def reset(self):
        """Reset all tracking state."""
        self.history.clear()
        self.total_surprises = 0
        self.total_predictions = 0
        self.surprise_rate_history.clear()
