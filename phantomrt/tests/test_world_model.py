"""Tests for the World Model."""
import pytest
import torch
from atlas.core.world_model import WorldModel


class TestWorldModel:
    @pytest.fixture
    def model(self):
        return WorldModel(
            obs_dim=32,
            action_dim=4,
            latent_dim=64,
            hidden_dim=128,
        )
    
    def test_encode(self, model):
        obs = torch.randn(4, 32)
        mean, log_var, z = model.encode(obs)
        
        assert mean.shape == (4, 64)
        assert log_var.shape == (4, 64)
        assert z.shape == (4, 64)
    
    def test_predict(self, model):
        latent = torch.randn(4, 64)
        obs_pred = model.predict(latent)
        
        assert obs_pred.shape == (4, 32)
    
    def test_forward_pass(self, model):
        obs = torch.randn(4, 32)
        output = model(obs)
        
        assert output.latent_state.shape == (4, 64)
        assert output.reconstructed_obs.shape == (4, 32)
        assert output.total_loss.item() > 0
    
    def test_forward_with_actions(self, model):
        obs = torch.randn(4, 32)
        actions = torch.randn(4, 5, 4)
        
        output = model(obs, actions)
        
        assert output.total_loss.item() > 0
    
    def test_imagine(self, model):
        obs = torch.randn(2, 32)
        actions = torch.randn(2, 10, 4)
        
        rollout = model.imagine(obs, actions)
        
        assert rollout.trajectory.shape[0] == 2
        assert rollout.trajectory.shape[2] == 64
        assert rollout.predicted_observations.shape[2] == 32
    
    def test_step_dynamics(self, model):
        latent = torch.randn(4, 64)
        action = torch.randn(4, 4)
        
        next_state = model.step_dynamics(latent, action)
        
        assert next_state.shape == (4, 64)
    
    def test_surprise_detection(self, model):
        obs = torch.randn(4, 32)
        output = model(obs)
        
        assert isinstance(output.is_surprising, bool)
        assert output.surprise_score >= 0
    
    def test_reward_predictor(self, model):
        latent = torch.randn(4, 64)
        reward = model.compute_reward(latent)
        
        assert reward.shape == (4, 1)
