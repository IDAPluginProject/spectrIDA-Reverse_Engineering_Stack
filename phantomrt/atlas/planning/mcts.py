"""
Monte Carlo Tree Search in Imagination

This is how the world model PLANS — by simulating
multiple possible futures in its imagination
and picking the best one.

Process:
1. Start from current state
2. Sample random action sequences
3. Imagine the future for each sequence
4. Score each imagined future by goal satisfaction
5. Pick the action sequence with the highest score
6. Execute the first action, then re-plan (receding horizon)

This is conceptually similar to how AlphaGo works,
but instead of a game board, we're planning in reality.
"""

import torch
import numpy as np
from typing import Optional

from ..core.world_model import WorldModel
from .goal import Goal


class ImaginationPlanner:
    """
    Plans actions by running Monte Carlo simulations
    in the world model's imagination.
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int,
        num_simulations: int = 100,
        horizon: int = 20,
        exploration_noise: float = 0.3,
        discount_factor: float = 0.99,
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.horizon = horizon
        self.exploration_noise = exploration_noise
        self.discount_factor = discount_factor
        self.device = device
    
    def plan(
        self,
        observation: torch.Tensor,
        goal: Goal,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Plan the best action sequence by imagining many futures.
        
        Args:
            observation: [1, obs_dim] current observation
            goal: Goal object to evaluate futures against
            return_all: if True, return all trajectories for analysis
            
        Returns:
            best_action: [1, action_dim] the best action to take now
        """
        self.world_model.eval()
        
        batch_size = observation.shape[0]
        
        # Get current latent state (deterministic for planning)
        current_state = self.world_model.get_latent_representation(observation)
        # [batch, latent_dim]
        
        best_value = -float("inf")
        best_actions = None
        all_trajectories = []
        all_values = []
        
        with torch.no_grad():
            for _ in range(self.num_simulations):
                # Sample a random action sequence
                actions = torch.randn(
                    batch_size, self.horizon, self.action_dim,
                    device=self.device
                ) * self.exploration_noise
                
                # Imagine the future
                latent_trajectory = self.world_model.neural_ode(
                    current_state, actions
                )
                # [batch, horizon+1, latent_dim]
                
                # Score this imagined future
                value = self._evaluate_trajectory(latent_trajectory, goal)
                
                if return_all:
                    all_trajectories.append(latent_trajectory)
                    all_values.append(value)
                
                if value.mean() > best_value:
                    best_value = value.mean()
                    best_actions = actions
        
        # Return the first action from the best sequence
        best_action = best_actions[:, 0:1]  # [batch, 1, action_dim]
        
        if return_all:
            return best_action, {
                "all_trajectories": torch.stack(all_trajectories),
                "all_values": torch.stack(all_values),
                "best_value": best_value,
            }
        
        return best_action
    
    def plan_with_rollouts(
        self,
        observation: torch.Tensor,
        goal: Goal,
        num_rollouts: int = 50,
        re_plan_steps: int = 5,
    ) -> list[torch.Tensor]:
        """
        Receding horizon planning — plan, execute some steps, re-plan.
        
        Returns a list of actions to execute.
        """
        actions = []
        current_obs = observation
        
        for _ in range(re_plan_steps):
            action = self.plan(current_obs, goal)
            actions.append(action.squeeze(1))  # remove time dim
            
            # Simulate taking this action
            rollout = self.world_model.imagine(current_obs, action)
            next_latent = rollout.trajectory[:, 1]
            next_obs_pred = self.world_model.predict(next_latent)
            current_obs = next_obs_pred
        
        return actions
    
    def _evaluate_trajectory(
        self,
        latent_trajectory: torch.Tensor,
        goal: Goal,
    ) -> torch.Tensor:
        """
        Evaluate how good an imagined trajectory is.
        
        Considers:
        1. Goal satisfaction at the final state
        2. Smooth trajectory (penalize jerky motion)
        3. Stay within bounds
        """
        batch_size = latent_trajectory.shape[0]
        
        # 1. Goal satisfaction at final state
        final_state = latent_trajectory[:, -1]
        goal_score = goal.satisfaction_score(final_state)
        
        # 2. Trajectory smoothness (penalize large jumps)
        diffs = latent_trajectory[:, 1:] - latent_trajectory[:, :-1]
        smoothness_penalty = torch.mean(diffs ** 2) * 0.1
        
        # 3. Reward accumulation (decoded from latent states)
        decoded_obs = self.world_model.decoder(
            latent_trajectory.reshape(-1, latent_trajectory.shape[-1])
        ).reshape(latent_trajectory.shape[0], latent_trajectory.shape[1], -1)
        
        # Simple reward signal: penalize being far from goal
        # (More sophisticated reward predictors can be used)
        total_value = goal_score - smoothness_penalty
        
        return total_value
    
    def visualize_imagination(
        self,
        observation: torch.Tensor,
        goal: Goal,
        num_samples: int = 5,
    ) -> dict:
        """
        Generate multiple imagined trajectories for visualization.
        
        Returns:
            dict with trajectories, observations, and scores
        """
        self.world_model.eval()
        
        current_state = self.world_model.get_latent_representation(observation)
        
        results = {
            "trajectories": [],
            "observations": [],
            "scores": [],
            "actions": [],
        }
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample random actions
                actions = torch.randn(
                    1, self.horizon, self.action_dim,
                    device=self.device
                ) * self.exploration_noise
                
                # Imagine
                latent_traj = self.world_model.neural_ode(current_state, actions)
                
                # Decode to observations
                flat = latent_traj.reshape(-1, self.world_model.latent_dim)
                obs_traj = self.world_model.decoder(flat).reshape(
                    1, self.horizon + 1, -1
                )
                
                # Score
                score = self._evaluate_trajectory(latent_traj, goal)
                
                results["trajectories"].append(latent_traj)
                results["observations"].append(obs_traj)
                results["scores"].append(score)
                results["actions"].append(actions)
        
        return results
