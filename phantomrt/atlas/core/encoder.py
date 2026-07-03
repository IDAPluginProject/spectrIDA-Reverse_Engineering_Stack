"""
Encoder: Observation → Latent State

Compresses raw observations (images, vectors, etc.) into
a compact latent representation that captures the ESSENCE
of the current state.

Uses variational inference to output a probability distribution
over possible states, not just a single point. This lets the
model express UNCERTAINTY about what it's seeing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes observations into a latent state distribution.
    
    Input:  observation tensor [batch, obs_dim]
    Output: (mean, log_variance) each [batch, latent_dim]
    
    The latent state is sampled via the reparameterization trick:
        z = mean + std * epsilon, where epsilon ~ N(0, 1)
    
    This allows gradients to flow through the sampling process.
    """

    def __init__(self, obs_dim: int, latent_dim: int = 256, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 512]
        
        # Build encoder network
        layers = []
        prev_dim = obs_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output heads: mean and log_variance of latent distribution
        self.mean_head = nn.Linear(prev_dim, latent_dim)
        self.log_var_head = nn.Linear(prev_dim, latent_dim)
        
        # Initialize heads with small weights for stable training
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_var_head.weight, gain=0.1)
        nn.init.zeros_(self.log_var_head.bias)
    
    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation into latent distribution parameters.
        
        Args:
            observation: [batch, obs_dim] raw observation
            
        Returns:
            (mean, log_variance): each [batch, latent_dim]
        """
        features = self.feature_net(observation)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        # Clamp log_var to prevent numerical instability
        # Range: [-10, 2] → std range: [~0.00005, ~7.4]
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        
        return mean, log_var
    
    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from the latent distribution using reparameterization trick.
        
        z = μ + σ * ε,  where ε ~ N(0, I)
        
        This is differentiable — gradients flow through the sample
        back to the encoder parameters.
        """
        std = torch.exp(0.5 * log_var)  # convert log_var to std
        epsilon = torch.randn_like(std)  # random noise ~ N(0, 1)
        z = mean + std * epsilon
        return z
    
    def encode(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encoding: observation → (mean, log_var, sampled_state)
        
        Convenience method that does everything in one call.
        """
        mean, log_var = self.forward(observation)
        z = self.sample(mean, log_var)
        return mean, log_var, z
    
    def kl_divergence(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from the latent distribution to N(0, I).
        
        KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        
        This regularizes the latent space to stay close to a standard
        normal distribution. Prevents the model from cheating by
        encoding everything into degenerate distributions.
        """
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()
