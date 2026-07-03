"""
Main Training Loop for the World Model

The training process:
1. Collect experience from the environment
2. Encode observations into latent states
3. Roll out dynamics predictions
4. Compute losses (reconstruction + KL + dynamics)
5. Update model via gradient descent
6. Repeat, tracking surprise rate and loss curves

Key insight: the surprise detector modulates which
experiences get used for learning. High-surprise
experiences get more gradient updates.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..core.world_model import WorldModel
from ..environments.base import BaseEnvironment
from .losses import compute_total_loss


class ReplayBuffer:
    """Stores past experiences for training."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, observation, action, reward, next_observation, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (observation, action, reward, next_observation, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> list:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return list(zip(*batch))  # unzip into separate lists
    
    def __len__(self):
        return len(self.buffer)


class Trainer:
    """
    Handles the full training pipeline for the world model.
    """

    def __init__(
        self,
        world_model: WorldModel,
        environment: BaseEnvironment,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        rollout_length: int = 20,
        kl_weight: float = 0.01,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "experiments",
        device: str = "cpu",
    ):
        self.world_model = world_model.to(device)
        self.environment = environment
        self.device = device
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.kl_weight = kl_weight
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-5
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "surprise_rate": [],
            "learning_rate": [],
        }
    
    def collect_experience(self, num_steps: int = 100):
        """Run the environment and store transitions."""
        obs = self.environment.reset()
        
        for _ in range(num_steps):
            # Random action for now (will use planner later)
            action = np.random.randn(self.environment.get_action_dim()).astype(np.float32)
            action = np.clip(action, -1, 1)
            
            next_obs, reward, done, info = self.environment.step(action)
            
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            
            if done:
                obs = self.environment.reset()
            else:
                obs = next_obs
    
    def train_step(self) -> dict:
        """Single training step on a batch from replay buffer."""
        self.world_model.train()
        
        # Sample batch
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        observations, actions, rewards, next_observations, dones = batch
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        next_obs_tensor = torch.FloatTensor(np.array(next_observations)).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        output = self.world_model(obs_tensor, action_tensor.unsqueeze(1))
        
        # Compute dynamics loss: predict next observation from current + action
        _, _, latent_state = self.world_model.encode(obs_tensor)
        predicted_next_state = self.world_model.step_dynamics(latent_state, action_tensor)
        predicted_next_obs = self.world_model.predict(predicted_next_state)
        dynamics_loss = nn.functional.mse_loss(predicted_next_obs, next_obs_tensor)
        
        # Total loss
        total_loss = (
            output.reconstruction_loss 
            + self.kl_weight * output.kl_loss
            + 0.1 * dynamics_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "recon_loss": output.reconstruction_loss.item(),
            "kl_loss": output.kl_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "surprise_score": output.surprise_score,
            "is_surprising": output.is_surprising,
        }
    
    def train_epoch(self, steps_per_epoch: int = 100) -> dict:
        """Full training epoch."""
        self.epoch += 1
        
        # Collect fresh experience
        self.collect_experience(num_steps=steps_per_epoch)
        
        # Multiple training steps per epoch
        epoch_losses = []
        num_train_steps = min(steps_per_epoch, len(self.replay_buffer) // self.batch_size)
        
        for _ in range(num_train_steps):
            result = self.train_step()
            if result is not None:
                epoch_losses.append(result)
        
        if not epoch_losses:
            return {}
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0]:
            if key == "is_surprising":
                avg_losses[key] = sum(r[key] for r in epoch_losses) / len(epoch_losses)
            else:
                avg_losses[key] = sum(r[key] for r in epoch_losses) / len(epoch_losses)
        
        # Update history
        self.history["train_loss"].append(avg_losses["total_loss"])
        self.history["recon_loss"].append(avg_losses["recon_loss"])
        self.history["kl_loss"].append(avg_losses["kl_loss"])
        self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
        
        # Update learning rate
        self.scheduler.step()
        
        # Track surprise rate
        surprise_stats = self.world_model.get_surprise_stats()
        self.history["surprise_rate"].append(surprise_stats["surprise_rate"])
        
        return avg_losses
    
    def train(self, num_epochs: int = 500, steps_per_epoch: int = 100):
        """Full training loop."""
        print(f"[ATLAS] PROJECT ATLAS — Training Phase 1")
        print(f"   Environment: {self.environment.__class__.__name__}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   {'='*50}")
        
        for epoch in range(num_epochs):
            losses = self.train_epoch(steps_per_epoch)
            
            if losses and epoch % 10 == 0:
                surprise = losses.get("is_surprising", 0)
                surprise_emoji = "[!]" if surprise else "[OK]"
                
                print(
                    f"  Epoch {epoch:4d} | "
                    f"Loss: {losses['total_loss']:.4f} | "
                    f"Recon: {losses['recon_loss']:.4f} | "
                    f"KL: {losses['kl_loss']:.4f} | "
                    f"Dyn: {losses['dynamics_loss']:.4f} | "
                    f"{surprise_emoji}"
                )
            
            # Checkpoint
            if epoch % 50 == 0 and losses:
                self.save_checkpoint(f"epoch_{epoch}.pt")
                
                if losses["total_loss"] < self.best_loss:
                    self.best_loss = losses["total_loss"]
                    self.save_checkpoint("best.pt")
        
        # Save final
        self.save_checkpoint("final.pt")
        self.save_history()
        
        print(f"\n{'='*50}")
        print(f"[OK] Training complete! Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.world_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
    
    def save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def train(config: dict, environment: BaseEnvironment, world_model: WorldModel):
    """Convenience function to start training."""
    trainer = Trainer(
        world_model=world_model,
        environment=environment,
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 64),
        rollout_length=config.get("rollout_length", 20),
        kl_weight=config.get("kl_weight", 0.01),
        gradient_clip=config.get("gradient_clip", 1.0),
        checkpoint_dir=config.get("save_dir", "experiments"),
        device=config.get("device", "cpu"),
    )
    trainer.train(
        num_epochs=config.get("max_epochs", 500),
        steps_per_epoch=config.get("steps_per_epoch", 100),
    )
    return trainer
