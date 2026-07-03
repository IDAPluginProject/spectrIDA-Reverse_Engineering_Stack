"""
Decoder: Latent State → Predicted Observation

The mirror of the encoder. Takes the internal
representation and reconstructs what the
observation SHOULD look like.

If the decoder can reconstruct reality from the
latent state, it means the latent state captured
the important information — the model UNDERSTOOD
what it was seeing.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decodes latent state back into observation space.
    
    Input:  latent_state [batch, latent_dim]
    Output: predicted_observation [batch, obs_dim]
    """

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dims: list = None):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 512]
        
        layers = []
        prev_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
            ])
            prev_dim = h_dim
        
        # Final layer: map back to observation space
        layers.append(nn.Linear(prev_dim, obs_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer for reasonable initial reconstructions
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state into predicted observation.
        
        Args:
            latent_state: [batch, latent_dim]
            
        Returns:
            predicted_obs: [batch, obs_dim]
        """
        return self.net(latent_state)
