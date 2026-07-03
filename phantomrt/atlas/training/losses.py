"""
Training Losses for the World Model

The total loss is a combination of:
1. Reconstruction loss: can the decoder recreate the input?
2. KL divergence: is the latent space well-organized?
3. Dynamics loss: does the model predict the future correctly?
4. Surprise-weighted loss: focus learning on surprising events
"""

import torch
import torch.nn.functional as F


def compute_reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE reconstruction loss.
    
    Measures how well the encoder→decoder pipeline
    can reconstruct the original observation.
    """
    return F.mse_loss(predicted, target)


def compute_kl_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from N(0, I).
    
    Regularizes the latent space to be smooth and organized.
    Without this, the encoder could memorize inputs with
    arbitrary latent representations.
    """
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
    return kl.mean()


def compute_dynamics_loss(
    world_model,
    initial_obs: torch.Tensor,
    actions: torch.Tensor,
    future_observations: torch.Tensor,
) -> torch.Tensor:
    """
    Dynamics prediction loss.
    
    Roll out the world model's predictions and compare
    against what actually happened.
    
    This trains the dynamics model to be accurate.
    """
    # Encode initial state
    _, _, initial_state = world_model.encode(initial_obs)
    
    # Roll out dynamics
    latent_traj = world_model.neural_ode(initial_state, actions)
    # [batch, steps+1, latent_dim]
    
    # Decode predicted observations
    batch_size, num_steps, latent_dim = latent_traj.shape
    flat = latent_traj.reshape(-1, latent_dim)
    predicted_obs = world_model.decoder(flat)
    predicted_obs = predicted_obs.reshape(batch_size, num_steps, -1)
    
    # Compare with actual future observations
    # future_observations: [batch, steps, obs_dim]
    loss = F.mse_loss(predicted_obs[:, 1:], future_observations)
    
    return loss


def compute_total_loss(
    world_model,
    observation: torch.Tensor,
    actions: torch.Tensor = None,
    future_observations: torch.Tensor = None,
    kl_weight: float = 0.01,
    dynamics_weight: float = 0.1,
) -> dict:
    """
    Compute all losses and return a dict.
    
    This is the main loss function used during training.
    """
    # Forward pass through world model
    output = world_model(observation, actions)
    
    losses = {
        "reconstruction": output.reconstruction_loss,
        "kl": output.kl_loss,
        "total": output.total_loss,
        "surprise_score": output.surprise_score,
        "is_surprising": output.is_surprising,
    }
    
    # Add dynamics loss if we have future observations
    if future_observations is not None and actions is not None:
        dyn_loss = compute_dynamics_loss(world_model, observation, actions, future_observations)
        losses["dynamics"] = dyn_loss
        losses["total"] = losses["total"] + dynamics_weight * dyn_loss
    
    return losses
