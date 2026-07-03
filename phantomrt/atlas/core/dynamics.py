"""
Dynamics Model: The Physics Engine

This is the HEART of the world model.
It learns HOW THE WORLD EVOLVES over time.

Instead of discrete layers, we use a Neural ODE —
a continuous differential equation that describes
how the state changes:

    dx/dt = f(x, action)

This is fundamentally different from transformers:
- Continuous, not discrete
- Learns dynamics, not patterns
- Can simulate arbitrary time horizons
- Naturally handles variable-speed events
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class DynamicsFunction(nn.Module):
    """
    Neural network that defines the dynamics: dx/dt = f(x, action)
    
    Input:  (state, action) concatenated
    Output: derivative dx/dt (rate of change of state)
    
    The network learns the PHYSICS of the environment —
    how objects move, interact, and change over time.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build dynamics network
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else state_dim
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            # Skip connection every 2 layers for gradient flow
            if i > 0 and i % 2 == 0:
                layers.append(SkipConnection(hidden_dim))
        
        # Final layer outputs the derivative
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize last layer with small weights
        # (prevents explosive dynamics at the start)
        last_layer = self.net[-1]
        nn.init.xavier_uniform_(last_layer.weight, gain=0.01)
        nn.init.zeros_(last_layer.bias)
    
    def forward(self, state: torch.Tensor, t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative dx/dt.
        
        Args:
            state: [batch, state_dim] current state
            t: scalar or [batch] current time (needed for ODE solver)
            action: [batch, action_dim] action being taken
            
        Returns:
            dx_dt: [batch, state_dim] rate of change of state
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Compute derivative
        dx_dt = self.net(x)
        
        # Optional: clamp derivatives to prevent explosion
        dx_dt = torch.clamp(dx_dt, min=-10.0, max=10.0)
        
        return dx_dt


class SkipConnection(nn.Module):
    """Simple skip connection for better gradient flow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return x + self.norm(x)


class NeuralODE(nn.Module):
    """
    Wraps a dynamics function with an ODE solver.
    
    This lets us:
    1. Integrate forward in time (simulate the future)
    2. Backpropagate through the ODE solver (adjoint method)
    3. Use adaptive step sizing (accuracy + efficiency)
    """

    def __init__(
        self,
        dynamics_fn: DynamicsFunction,
        solver: str = "dopri5",
        dt: float = 0.05,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        self.solver = solver
        self.dt = dt
        self.rtol = rtol
        self.atol = atol
    
    def forward(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        time_horizon: float = None,
    ) -> torch.Tensor:
        """
        Roll out the dynamics forward in time.
        
        Args:
            initial_state: [batch, state_dim] starting state
            actions: [batch, num_steps, action_dim] action sequence
                     OR [batch, action_dim] for single constant action
            time_horizon: total simulation time (if None, len(actions) * dt)
            
        Returns:
            trajectory: [batch, num_steps+1, state_dim]
                       (includes initial state at t=0)
        """
        batch_size = initial_state.shape[0]
        
        # Handle single action case
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        
        num_steps = actions.shape[1]
        
        if time_horizon is None:
            time_horizon = num_steps * self.dt
        
        t_span = torch.linspace(0, time_horizon, num_steps + 1, device=initial_state.device)
        
        # Interpolate actions at solver timesteps
        def dynamics_with_action(t, state):
            # Find the closest action for current time
            action_idx = torch.clamp(
                (t / self.dt).long(), 
                min=0, 
                max=num_steps - 1
            )
            # Get actions for this batch
            action = actions[:, action_idx]  # [batch, action_dim]
            return self.dynamics_fn(state, t, action)
        
        # Solve the ODE
        trajectory = odeint(
            dynamics_with_action,
            initial_state,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        
        # trajectory shape: [num_timesteps, batch, state_dim]
        # Transpose to: [batch, num_timesteps, state_dim]
        trajectory = trajectory.transpose(0, 1)
        
        return trajectory
    
    def single_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float = None,
    ) -> torch.Tensor:
        """
        Single-step prediction: state(t) → state(t+dt)
        
        Faster than full rollout for training.
        """
        if dt is None:
            dt = self.dt
        
        t_span = torch.tensor([0.0, dt], device=state.device)
        
        def dynamics(t, s):
            return self.dynamics_fn(s, t, action)
        
        # Solve for one step
        result = odeint(
            dynamics,
            state,
            t_span,
            method="euler",  # fast for single step
            options={"step_size": dt},
        )
        
        # Return final state: [batch, state_dim]
        return result[-1]
