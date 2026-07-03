"""
Grid World Environment

A simple 2D grid where an agent navigates to reach a goal.
Objects can be placed on the grid and have basic properties.

This is the TESTBED for Phase 1 — simple enough to learn,
complex enough to test understanding.

Observation space: [agent_x, agent_y, goal_x, goal_y, obj1_x, obj1_y, obj1_type, ...]
Action space: [dx, dy] continuous movement
"""

import numpy as np
from typing import Optional
from .base import BaseEnvironment


class GridWorld(BaseEnvironment):
    """
    2D grid world with:
    - Agent (orange) that moves around
    - Goal (green) to reach
    - Obstacles (red) that block movement
    - Collectibles (blue) that give reward
    """

    def __init__(
        self,
        grid_size: int = 8,
        num_obstacles: int = 3,
        num_collectibles: int = 2,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_collectibles = num_collectibles
        self.max_steps = max_steps
        
        self.rng = np.random.RandomState(seed)
        
        # State
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.collectibles = None
        self.collected = None
        self.steps = 0
        
        # Observation: [agent_x, agent_y, goal_x, goal_y, 
        #               obs1_x, obs1_y, obs1_exists,
        #               obs2_x, obs2_y, obs2_exists, ...]
        # For each object: x, y, exists (3 values)
        self._obs_dim = 4 + 3 * (num_obstacles + num_collectibles)
        self._action_dim = 2  # dx, dy
    
    def get_observation_dim(self) -> int:
        return self._obs_dim
    
    def get_action_dim(self) -> int:
        return self._action_dim
    
    def reset(self) -> np.ndarray:
        """Reset to a random configuration."""
        self.steps = 0
        self.collected = [False] * self.num_collectibles
        
        # Place agent at random position
        self.agent_pos = self.rng.randint(0, self.grid_size, size=2).astype(float)
        
        # Place goal at random position (not on agent)
        while True:
            self.goal_pos = self.rng.randint(0, self.grid_size, size=2).astype(float)
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break
        
        # Place obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = self.rng.randint(0, self.grid_size, size=2).astype(float)
                if (not np.array_equal(pos, self.agent_pos) and 
                    not np.array_equal(pos, self.goal_pos) and
                    not any(np.array_equal(pos, o) for o in self.obstacles)):
                    self.obstacles.append(pos)
                    break
        
        # Place collectibles
        self.collectibles = []
        for _ in range(self.num_collectibles):
            while True:
                pos = self.rng.randint(0, self.grid_size, size=2).astype(float)
                if (not np.array_equal(pos, self.agent_pos) and 
                    not np.array_equal(pos, self.goal_pos) and
                    not any(np.array_equal(pos, o) for o in self.obstacles) and
                    not any(np.array_equal(pos, c) for c in self.collectibles)):
                    self.collectibles.append(pos)
                    break
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take a movement action."""
        self.steps += 1
        
        # Clip and apply action
        action = np.clip(action, -1.0, 1.0)
        new_pos = self.agent_pos + action
        
        # Clip to grid bounds
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        
        # Check obstacle collision
        hit_obstacle = False
        for obs_pos in self.obstacles:
            if self._check_collision(new_pos, obs_pos):
                hit_obstacle = True
                new_pos = self.agent_pos.copy()  # don't move
                break
        
        self.agent_pos = new_pos
        
        # Compute reward
        reward = 0.0
        
        # Check goal reached
        reached_goal = self._check_reached(self.agent_pos, self.goal_pos)
        if reached_goal:
            reward += 10.0
        
        # Check collectible pickup
        for i, coll_pos in enumerate(self.collectibles):
            if not self.collected[i] and self._check_reached(self.agent_pos, coll_pos):
                self.collected[i] = True
                reward += 1.0
        
        # Small step penalty (encourages efficiency)
        reward -= 0.01
        
        # Collision penalty
        if hit_obstacle:
            reward -= 0.5
        
        # Check done
        done = reached_goal or self.steps >= self.max_steps
        
        info = {
            "reached_goal": reached_goal,
            "hit_obstacle": hit_obstacle,
            "steps": self.steps,
            "collected": sum(self.collected),
        }
        
        return self._get_observation(), reward, done, info
    
    def _check_collision(self, pos1: np.ndarray, pos2: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if two positions are close enough to collide."""
        return np.linalg.norm(pos1 - pos2) < threshold
    
    def _check_reached(self, pos: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if position has reached the target."""
        return np.linalg.norm(pos - target) < threshold
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state."""
        obs = []
        
        # Agent and goal
        obs.extend(self.agent_pos / self.grid_size)  # normalize to [0, 1]
        obs.extend(self.goal_pos / self.grid_size)
        
        # Obstacles
        for obs_pos in self.obstacles:
            obs.extend(obs_pos / self.grid_size)
            obs.append(1.0)  # exists
        
        # Collectibles
        for i, coll_pos in enumerate(self.collectibles):
            obs.extend(coll_pos / self.grid_size)
            obs.append(0.0 if self.collected[i] else 1.0)  # exists (0 if collected)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self) -> np.ndarray:
        """Render grid as RGB image."""
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * 240  # light gray background
        
        # Draw obstacles (red)
        for obs_pos in self.obstacles:
            x, y = int(obs_pos[0]), int(obs_pos[1])
            img[y, x] = [220, 50, 50]
        
        # Draw collectibles (blue) - only if not collected
        for i, coll_pos in enumerate(self.collectibles):
            if not self.collected[i]:
                x, y = int(coll_pos[0]), int(coll_pos[1])
                img[y, x] = [50, 50, 220]
        
        # Draw goal (green)
        x, y = int(self.goal_pos[0]), int(self.goal_pos[1])
        img[y, x] = [50, 200, 50]
        
        # Draw agent (orange)
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        img[y, x] = [255, 165, 0]
        
        return img
    
    def get_state(self) -> dict:
        """Get full state for debugging."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "obstacles": [o.copy() for o in self.obstacles],
            "collectibles": [c.copy() for c in self.collectibles],
            "collected": self.collected.copy(),
            "steps": self.steps,
        }
