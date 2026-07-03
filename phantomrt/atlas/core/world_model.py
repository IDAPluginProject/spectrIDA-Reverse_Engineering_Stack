"""
World Model: The Complete Brain

Combines encoder, dynamics, and decoder into one coherent system
that can:
1. Encode observations into understanding
2. Simulate the future in imagination
3. Predict what it would see
4. Detect surprises and learn from them
5. Plan actions by imagining outcomes

This is the central class that everything else interacts with.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder
from .dynamics import DynamicsFunction, NeuralODE
from .surprise import SurpriseDetector


@dataclass
class WorldModelOutput:
    """Container for world model outputs."""
    # Current state
    latent_state: torch.Tensor          # [batch, latent_dim]
    encoder_mean: torch.Tensor          # [batch, latent_dim]
    encoder_log_var: torch.Tensor       # [batch, latent_dim]
    
    # Reconstruction
    reconstructed_obs: torch.Tensor     # [batch, obs_dim]
    
    # Losses
    reconstruction_loss: torch.Tensor   # scalar
    kl_loss: torch.Tensor               # scalar
    surprise_loss: torch.Tensor         # scalar
    total_loss: torch.Tensor            # scalar
    
    # Surprise info
    is_surprising: bool
    surprise_score: float


@dataclass
class RolloutOutput:
    """Container for imagined trajectory outputs."""
    trajectory: torch.Tensor            # [batch, steps+1, latent_dim]
    predicted_observations: torch.Tensor # [batch, steps+1, obs_dim]
    rewards: Optional[torch.Tensor]     # [batch, steps+1] if reward predictor exists


class WorldModel(nn.Module):
    """
    The complete world model brain.
    
    Architecture:
        observation → [Encoder] → latent_state
                                    ↓
                            [Neural ODE Dynamics] → future_state
                                    ↓
                            [Decoder] → predicted_observation
                                    ↓
                            [Surprise] → learn or confirm
    
    Key capabilities:
        - encode(): understand what we're seeing
        - imagine(): simulate future states
        - predict(): see what we'd observe from a state
        - plan(): find best action by imagining outcomes
        - learn(): update from surprising experiences
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        dynamics_layers: int = 3,
        dynamics_solver: str = "dopri5",
        dynamics_dt: float = 0.05,
        dropout: float = 0.1,
        surprise_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [hidden_dim, hidden_dim]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [hidden_dim, hidden_dim]
        
        # === Core Components ===
        self.encoder = Encoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )
        
        self.dynamics_fn = DynamicsFunction(
            state_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=dynamics_layers,
        )
        
        self.neural_ode = NeuralODE(
            dynamics_fn=self.dynamics_fn,
            solver=dynamics_solver,
            dt=dynamics_dt,
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dims=decoder_hidden_dims,
        )
        
        # === Surprise System ===
        self.surprise_detector = SurpriseDetector(
            initial_threshold=surprise_threshold,
        )
        
        # === Reward Predictor (for planning) ===
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def encode(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation into latent state.
        
        Args:
            observation: [batch, obs_dim]
            
        Returns:
            (mean, log_var, sampled_state): each [batch, latent_dim]
        """
        return self.encoder.encode(observation)
    
    def predict(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state into predicted observation.
        
        Args:
            latent_state: [batch, latent_dim]
            
        Returns:
            predicted_observation: [batch, obs_dim]
        """
        return self.decoder(latent_state)
    
    def imagine(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> RolloutOutput:
        """
        Imagine a future by rolling out dynamics in latent space.
        
        This is the CORE capability — the model simulates
        what WOULD happen if it took certain actions,
        WITHOUT touching the real world.
        
        Args:
            initial_observation: [batch, obs_dim]
            actions: [batch, num_steps, action_dim]
            
        Returns:
            RolloutOutput with trajectory and predicted observations
        """
        # Encode current observation
        _, _, initial_state = self.encode(initial_observation)
        
        # Roll out dynamics in latent space
        latent_trajectory = self.neural_ode(initial_state, actions)
        # shape: [batch, num_steps+1, latent_dim]
        
        # Decode each state into predicted observations
        batch_size, num_timesteps, _ = latent_trajectory.shape
        flat_states = latent_trajectory.reshape(-1, self.latent_dim)
        predicted_obs = self.decoder(flat_states)
        predicted_obs = predicted_obs.reshape(batch_size, num_timesteps, self.obs_dim)
        
        # Predict rewards for each state
        flat_rewards = self.reward_predictor(flat_states)
        rewards = flat_rewards.reshape(batch_size, num_timesteps, 1)
        
        return RolloutOutput(
            trajectory=latent_trajectory,
            predicted_observations=predicted_obs,
            rewards=rewards,
        )
    
    def step_dynamics(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step dynamics prediction.
        
        Args:
            latent_state: [batch, latent_dim]
            action: [batch, action_dim]
            
        Returns:
            next_state: [batch, latent_dim]
        """
        return self.neural_ode.single_step(latent_state, action)
    
    def forward(
        self,
        observation: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rollout_steps: int = 10,
    ) -> WorldModelOutput:
        """
        Full forward pass: encode → reconstruct → compute losses.
        
        If actions are provided, also rolls out dynamics and
        computes prediction losses.
        
        Args:
            observation: [batch, obs_dim]
            actions: optional [batch, rollout_steps, action_dim]
            rollout_steps: how many steps to predict
            
        Returns:
            WorldModelOutput with all losses and info
        """
        # 1. Encode
        mean, log_var, latent_state = self.encode(observation)
        
        # 2. Reconstruct
        reconstructed = self.predict(latent_state)
        
        # 3. Compute reconstruction loss
        recon_loss = F.mse_loss(reconstructed, observation)
        
        # 4. KL divergence (regularize latent space)
        kl_loss = self.encoder.kl_divergence(mean, log_var)
        
        # 5. Surprise detection
        surprise_loss, is_surprising = self.surprise_detector.compute_surprise(
            observation, reconstructed
        )
        
        # 6. Total loss (weighted combination)
        total_loss = recon_loss + 0.01 * kl_loss
        
        # If actions provided, add dynamics prediction loss
        if actions is not None:
            dynamics_loss = self._compute_dynamics_loss(observation, actions)
            total_loss = total_loss + 0.1 * dynamics_loss
        
        return WorldModelOutput(
            latent_state=latent_state,
            encoder_mean=mean,
            encoder_log_var=log_var,
            reconstructed_obs=reconstructed,
            reconstruction_loss=recon_loss,
            kl_loss=kl_loss,
            surprise_loss=surprise_loss,
            total_loss=total_loss,
            is_surprising=is_surprising,
            surprise_score=surprise_loss.item(),
        )
    
    def _compute_dynamics_loss(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dynamics prediction loss.
        
        Roll out the model's predictions and compare against
        what actually happened.
        """
        # Encode initial state
        _, _, initial_state = self.encode(observation)
        
        # Roll out dynamics
        latent_trajectory = self.neural_ode(initial_state, actions)
        
        # For each step, reconstruct and compare with next observation
        # (We don't have future observations here in the simplified case,
        #  so we use self-consistency: predicted future should predict itself)
        total_loss = torch.tensor(0.0, device=observation.device)
        
        for t in range(latent_trajectory.shape[1] - 1):
            current_state = latent_trajectory[:, t]
            next_state = latent_trajectory[:, t + 1]
            action = actions[:, t]
            
            # Predict next state from current
            predicted_next = self.neural_ode.single_step(current_state, action)
            
            # Loss: predicted next should match actual next
            step_loss = F.mse_loss(predicted_next, next_state.detach())
            total_loss = total_loss + step_loss
        
        return total_loss / max(latent_trajectory.shape[1] - 1, 1)
    
    def get_latent_representation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get the deterministic latent state (mean, no sampling).
        
        Useful for planning and visualization.
        """
        mean, _ = self.encoder(observation)
        return mean
    
    def compute_reward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Predict reward for a given state."""
        return self.reward_predictor(latent_state)
    
    def get_surprise_stats(self) -> dict:
        """Get surprise detection statistics."""
        return self.surprise_detector.get_stats()
