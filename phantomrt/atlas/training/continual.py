"""
Catastrophic Forgetting Prevention

Multiple strategies to protect old knowledge
while learning new things.

The goal: model gets SMARTER over time,
not just different.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional
import random


class ExperienceReplay:
    """
    Store old experiences and replay them during training.
    
    The most proven defense against catastrophic forgetting.
    Like how your brain replays old memories during sleep.
    """
    
    def __init__(self, capacity: int = 50000, priority_alpha: float = 0.6):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        
        # Storage
        self.buffer = []
        self.priorities = []
        self.position = 0
        
        # Statistics
        self.total_added = 0
    
    def add(self, observation, priority: float = 1.0):
        """Add an experience with priority score."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(observation)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = observation
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1
    
    def sample(self, batch_size: int, old_ratio: float = 0.3) -> list:
        """
        Sample a batch with a mix of old and new experiences.
        
        old_ratio: fraction of batch from OLD experiences
        (ensures old knowledge gets rehearsed)
        """
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Prioritized sampling for old experiences
        old_count = int(batch_size * old_ratio)
        new_count = batch_size - old_count
        
        # Sample old experiences (with priority weighting)
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        old_indices = np.random.choice(
            len(self.buffer), size=old_count, replace=False, p=probs
        )
        old_samples = [self.buffer[i] for i in old_indices]
        
        # Sample recent experiences
        recent_start = max(0, len(self.buffer) - 1000)
        recent_indices = random.sample(
            range(recent_start, len(self.buffer)),
            min(new_count, len(self.buffer) - recent_start)
        )
        new_samples = [self.buffer[i] for i in recent_indices]
        
        return old_samples + new_samples
    
    def __len__(self):
        return len(self.buffer)


class ElasticWeightConsolidation:
    """
    EWC: Protects important weights from being overwritten.
    
    Key idea: some weights are IMPORTANT for old tasks.
    When learning new tasks, penalize changes to those weights.
    
    Math:
      L_total = L_new + λ × Σ F_i × (θ_i - θ*_i)²
      
      F_i = Fisher information (how important is weight i?)
      θ*_i = original weight value
      λ = protection strength
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store reference weights
        self.reference_weights = {}
        for name, param in model.named_parameters():
            self.reference_weights[name] = param.data.clone()
        
        # Fisher information (importance of each weight)
        self.fisher_information = {}
        self._compute_fisher()
    
    def _compute_fisher(self, num_samples: int = 100):
        """Compute Fisher information for each weight."""
        # Initialize fisher to zeros
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param.data)
        
        # Estimate Fisher from data
        self.model.eval()
        for _ in range(num_samples):
            # Forward pass with random data
            dummy = torch.randn(1, self.model.obs_dim)
            output = self.model(dummy)
            
            # Compute gradient of log-likelihood
            loss = output.total_loss
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
        
        # Average
        for name in self.fisher_information:
            self.fisher_information[name] /= num_samples
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty.
        
        High penalty for changing important weights.
        Low penalty for changing unimportant weights.
        """
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                reference = self.reference_weights[name]
                
                # Quadratic penalty: Σ F_i × (θ_i - θ*_i)²
                penalty += (fisher * (param - reference) ** 2).sum()
        
        return self.lambda_ewc * penalty
    
    def update_reference(self):
        """Update reference weights after learning new task."""
        for name, param in self.model.named_parameters():
            self.reference_weights[name] = param.data.clone()
        
        # Recompute Fisher
        self._compute_fisher()


class KnowledgeDistillation:
    """
    Distill knowledge from old model into new model.
    
    Keep a "teacher" copy of the old model.
    When training new knowledge, also ensure the new model
    produces similar outputs to the old model.
    
    This preserves BEHAVIOR even if weights change.
    """
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 2.0, alpha: float = 0.5):
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha  # balance new vs old knowledge
    
    def distillation_loss(self, student_output, new_loss):
        """
        Combined loss: learn new stuff + stay similar to teacher.
        
        L_total = α × L_new + (1-α) × L_distill
        """
        with torch.no_grad():
            teacher_output = self.teacher(student_output["input"])
        
        # Distillation loss: student should match teacher's latent states
        student_latent = student_output["latent_state"]
        teacher_latent = teacher_output["latent_state"]
        
        # Soft matching in latent space
        distill_loss = F.mse_loss(student_latent, teacher_latent)
        
        # Combined loss
        total = self.alpha * new_loss + (1 - self.alpha) * distill_loss
        
        return total


class ProgressiveNeuralNetworks:
    """
    Add NEW capacity for new tasks without touching old weights.
    
    Instead of one network that learns everything,
    have a column for each task, with lateral connections.
    
    Old columns stay FROZEN.
    New columns learn new stuff.
    Connections let them share knowledge.
    """
    
    def __init__(self, base_model: nn.Module, task_embedding_dim: int = 32):
        self.base_model = base_model
        self.frozen_columns = []  # old task models (frozen)
        self.current_column = base_model
        self.task_embeddings = nn.Embedding(100, task_embedding_dim)  # up to 100 tasks
    
    def freeze_current(self):
        """Freeze current column (task is done learning)."""
        for param in self.current_column.parameters():
            param.requires_grad = False
        self.frozen_columns.append(self.current_column)
    
    def add_new_column(self, new_column: nn.Module):
        """Add a new column for a new task."""
        self.current_column = new_column
    
    def forward(self, x, task_id: int = 0):
        """
        Forward pass through frozen columns + current column.
        Frozen columns provide knowledge from old tasks.
        Current column learns new task.
        """
        outputs = []
        
        # Process through frozen columns
        for frozen in self.frozen_columns:
            with torch.no_grad():
                frozen_out = frozen(x)
                outputs.append(frozen_out.latent_state)
        
        # Process through current column
        current_out = self.current_column(x)
        outputs.append(current_out.latent_state)
        
        # Combine (simple concatenation + projection)
        combined = torch.cat(outputs, dim=-1)
        
        return combined


class ContinualLearningManager:
    """
    Manages all anti-forgetting strategies.
    
    Combines:
      1. Experience replay (rehearse old memories)
      2. EWC (protect important weights)
      3. Knowledge distillation (preserve behavior)
      4. Surprise-gated learning (only learn when surprised)
    """
    
    def __init__(self, model: nn.Module, replay_capacity: int = 50000):
        self.model = model
        
        # Components
        self.replay_buffer = ExperienceReplay(capacity=replay_capacity)
        self.ewc = ElasticWeightConsolidation(model)
        self.teacher = None  # set after initial training
        
        # Learning tracking
        self.task_boundaries = []
        self.current_task = 0
        self.total_steps = 0
        
        # Forgetting detection
        self.old_performance = {}
        self.performance_history = []
    
    def update(self, observation: torch.Tensor, loss: torch.Tensor):
        """
        Update model with anti-forgetting protection.
        
        The key insight: loss is MODIFIED to protect old knowledge.
        """
        # 1. Add to replay buffer (with priority from loss)
        self.replay_buffer.add(
            observation.detach(),
            priority=loss.item() + 1.0
        )
        
        # 2. Compute EWC penalty (protect old weights)
        ewc_penalty = self.ewc.penalty()
        
        # 3. Sample old experiences for rehearsal
        if len(self.replay_buffer) > 100:
            old_batch = self.replay_buffer.sample(32, old_ratio=0.3)
            
            # Forward pass on old experiences
            old_data = torch.cat(old_batch, dim=0)
            old_output = self.model(old_data)
            old_loss = old_output.total_loss
            
            # Combined loss: new + old rehearsal + EWC protection
            total_loss = loss + 0.3 * old_loss + ewc_penalty
        else:
            total_loss = loss + ewc_penalty
        
        # 4. Track forgetting
        self._check_forgotten()
        
        self.total_steps += 1
        
        return total_loss
    
    def checkpoint_task(self, task_name: str):
        """
        Call this when a task/binary analysis is complete.
        
        Snapshots model state and prepares for next task.
        """
        self.task_boundaries.append({
            "task": task_name,
            "step": self.total_steps,
            "performance": self._evaluate_current(),
        })
        
        # Save old performance for forgetting detection
        self.old_performance[task_name] = self._evaluate_current()
        
        # Update EWC reference (new baseline)
        self.ewc.update_reference()
        
        self.current_task += 1
        
        print(f"Task '{task_name}' checkpointed. "
              f"Performance: {self.old_performance[task_name]:.4f}")
    
    def _evaluate_current(self) -> float:
        """Quick evaluation of current model performance."""
        if len(self.replay_buffer) < 50:
            return 0.0
        
        batch = self.replay_buffer.sample(32)
        data = torch.cat(batch, dim=0)
        
        with torch.no_grad():
            output = self.model(data)
        
        return output.reconstruction_loss.item()
    
    def _check_forgotten(self):
        """Check if old knowledge is being forgotten."""
        if self.total_steps % 100 != 0:
            return
        
        current_perf = self._evaluate_current()
        self.performance_history.append(current_perf)
        
        if len(self.performance_history) > 10:
            recent = self.performance_history[-10:]
            older = self.performance_history[-20:-10] if len(self.performance_history) > 20 else self.performance_history[:10]
            
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older) if older else recent_avg
            
            # If performance dropped significantly, we're forgetting
            if recent_avg > older_avg * 1.2:  # loss went UP = forgetting
                print(f"[WARNING] Potential forgetting detected at step {self.total_steps}")
                print(f"  Recent loss: {recent_avg:.4f} vs Earlier: {older_avg:.4f}")
    
    def get_stats(self) -> dict:
        """Get forgetting prevention stats."""
        return {
            "total_steps": self.total_steps,
            "replay_buffer_size": len(self.replay_buffer),
            "tasks_completed": self.current_task,
            "task_boundaries": self.task_boundaries,
            "performance_trend": self.performance_history[-10:] if self.performance_history else [],
        }
