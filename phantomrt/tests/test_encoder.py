"""Tests for the Encoder module."""
import pytest
import torch
from atlas.core.encoder import Encoder


class TestEncoder:
    def test_output_shapes(self):
        obs_dim = 32
        latent_dim = 64
        batch_size = 8
        
        encoder = Encoder(obs_dim, latent_dim)
        obs = torch.randn(batch_size, obs_dim)
        
        mean, log_var = encoder(obs)
        
        assert mean.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)
    
    def test_reparameterization(self):
        obs_dim = 16
        latent_dim = 32
        batch_size = 4
        
        encoder = Encoder(obs_dim, latent_dim)
        obs = torch.randn(batch_size, obs_dim)
        
        mean, log_var = encoder(obs)
        z = encoder.sample(mean, log_var)
        
        assert z.shape == (batch_size, latent_dim)
        # Should be different each time (stochastic)
        z2 = encoder.sample(mean, log_var)
        assert not torch.allclose(z, z2)
    
    def test_encode_method(self):
        obs_dim = 20
        latent_dim = 40
        
        encoder = Encoder(obs_dim, latent_dim)
        obs = torch.randn(5, obs_dim)
        
        mean, log_var, z = encoder.encode(obs)
        
        assert mean.shape == (5, latent_dim)
        assert log_var.shape == (5, latent_dim)
        assert z.shape == (5, latent_dim)
    
    def test_kl_divergence(self):
        obs_dim = 16
        latent_dim = 32
        
        encoder = Encoder(obs_dim, latent_dim)
        mean = torch.zeros(4, latent_dim)
        log_var = torch.zeros(4, latent_dim)
        
        kl = encoder.kl_divergence(mean, log_var)
        
        # KL from N(0,I) to N(0,I) should be ~0
        assert kl.item() < 0.1
    
    def test_log_var_clamping(self):
        obs_dim = 16
        latent_dim = 32
        
        encoder = Encoder(obs_dim, latent_dim)
        obs = torch.randn(4, obs_dim)
        
        mean, log_var = encoder(obs)
        
        assert log_var.min() >= -10.0
        assert log_var.max() <= 2.0
    
    def test_differentiability(self):
        obs_dim = 16
        latent_dim = 32
        
        encoder = Encoder(obs_dim, latent_dim)
        obs = torch.randn(4, obs_dim, requires_grad=True)
        
        mean, log_var, z = encoder.encode(obs)
        loss = z.sum()
        loss.backward()
        
        assert obs.grad is not None
