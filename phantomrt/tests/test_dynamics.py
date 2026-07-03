"""Tests for the Dynamics module."""
import pytest
import torch
from atlas.core.dynamics import DynamicsFunction, NeuralODE


class TestDynamicsFunction:
    def test_output_shape(self):
        state_dim = 64
        action_dim = 8
        
        dynamics = DynamicsFunction(state_dim, action_dim)
        state = torch.randn(4, state_dim)
        t = torch.tensor(0.0)
        action = torch.randn(4, action_dim)
        
        dx_dt = dynamics(state, t, action)
        
        assert dx_dt.shape == (4, state_dim)
    
    def test_derivative_clamping(self):
        dynamics = DynamicsFunction(32, 4)
        state = torch.randn(4, 32) * 100  # large values
        t = torch.tensor(0.0)
        action = torch.randn(4, 4)
        
        dx_dt = dynamics(state, t, action)
        
        assert dx_dt.max() <= 10.0
        assert dx_dt.min() >= -10.0


class TestNeuralODE:
    def test_single_step(self):
        state_dim = 32
        action_dim = 4
        
        dynamics = DynamicsFunction(state_dim, action_dim)
        node = NeuralODE(dynamics, solver="euler", dt=0.1)
        
        state = torch.randn(2, state_dim)
        action = torch.randn(2, action_dim)
        
        next_state = node.single_step(state, action, dt=0.1)
        
        assert next_state.shape == (2, state_dim)
    
    def test_rollout_shape(self):
        state_dim = 32
        action_dim = 4
        num_steps = 10
        
        dynamics = DynamicsFunction(state_dim, action_dim)
        node = NeuralODE(dynamics, solver="euler", dt=0.05)
        
        initial_state = torch.randn(2, state_dim)
        actions = torch.randn(2, num_steps, action_dim)
        
        trajectory = node(initial_state, actions)
        
        # Should be [batch, num_steps+1, state_dim]
        assert trajectory.shape == (2, num_steps + 1, state_dim)
    
    def test_single_action_rollout(self):
        state_dim = 32
        action_dim = 4
        
        dynamics = DynamicsFunction(state_dim, action_dim)
        node = NeuralODE(dynamics, solver="euler", dt=0.05)
        
        initial_state = torch.randn(2, state_dim)
        action = torch.randn(2, action_dim)  # single action
        
        trajectory = node(initial_state, action, time_horizon=0.5)
        
        assert trajectory.dim() == 3
        assert trajectory.shape[-1] == state_dim
    
    def test_differentiability(self):
        state_dim = 16
        action_dim = 2
        
        dynamics = DynamicsFunction(state_dim, action_dim)
        node = NeuralODE(dynamics, solver="euler", dt=0.1)
        
        initial_state = torch.randn(1, state_dim, requires_grad=True)
        action = torch.randn(1, action_dim)
        
        trajectory = node(initial_state, action)
        loss = trajectory[:, -1].sum()
        loss.backward()
        
        assert initial_state.grad is not None
