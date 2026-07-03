"""Tests for the Planning module."""
import pytest
import torch
from atlas.core.world_model import WorldModel
from atlas.planning.mcts import ImaginationPlanner
from atlas.planning.goal import PositionGoal, CompositeGoal


class TestImaginationPlanner:
    @pytest.fixture
    def setup(self):
        model = WorldModel(obs_dim=32, action_dim=4, latent_dim=64, hidden_dim=128)
        planner = ImaginationPlanner(
            world_model=model,
            action_dim=4,
            num_simulations=10,
            horizon=5,
        )
        return planner, model
    
    def test_plan_returns_action(self, setup):
        planner, model = setup
        obs = torch.randn(1, 32)
        goal = PositionGoal(target_position=torch.randn(2))
        
        action = planner.plan(obs, goal)
        
        assert action.shape == (1, 1, 4)  # [batch, 1, action_dim]
    
    def test_plan_batch(self, setup):
        planner, model = setup
        obs = torch.randn(4, 32)
        goal = PositionGoal(target_position=torch.randn(2))
        
        action = planner.plan(obs, goal)
        
        assert action.shape == (4, 1, 4)
    
    def test_visualize_imagination(self, setup):
        planner, model = setup
        obs = torch.randn(1, 32)
        goal = PositionGoal(target_position=torch.randn(2))
        
        results = planner.visualize_imagination(obs, goal, num_samples=3)
        
        assert len(results["trajectories"]) == 3
        assert len(results["scores"]) == 3


class TestGoals:
    def test_position_goal(self):
        goal = PositionGoal(target_position=torch.tensor([5.0, 5.0]))
        
        # At target
        state_at = torch.tensor([5.0, 5.0, 0.0, 0.0])
        score_at = goal.satisfaction_score(state_at)
        assert score_at.item() > 0.9
        
        # Far from target
        state_far = torch.tensor([0.0, 0.0, 0.0, 0.0])
        score_far = goal.satisfaction_score(state_far)
        assert score_far.item() < 0.1
    
    def test_composite_goal(self):
        goal1 = PositionGoal(target_position=torch.tensor([5.0, 5.0]))
        goal2 = PositionGoal(target_position=torch.tensor([0.0, 0.0]))
        
        composite = CompositeGoal([goal1, goal2], weights=[0.5, 0.5])
        
        state = torch.tensor([2.5, 2.5, 0.0, 0.0])
        score = composite.satisfaction_score(state)
        
        assert 0.0 < score.item() < 1.0
