"""
2D Physics Environment

Simple Newtonian physics simulation with:
- Balls that move with velocity
- Gravity
- Elastic collisions between balls
- Wall bouncing

This tests whether the world model can learn ACTUAL PHYSICS —
conservation of momentum, gravity, collision response.

Observation: [ball1_x, ball1_y, ball1_vx, ball1_vy, ball1_r, ball1_color, ...]
Action: [force_x, force_y] applied to agent ball
"""

import numpy as np
from typing import Optional
from .base import BaseEnvironment


class Ball:
    """Simple physics ball."""
    
    def __init__(self, x, y, vx=0, vy=0, radius=0.5, mass=1.0, color=0, fixed=False):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([vx, vy], dtype=np.float64)
        self.radius = radius
        self.mass = mass
        self.color = color  # 0=agent, 1=obstacle, 2=collectible
        self.fixed = fixed  # fixed balls don't move


class Physics2DEnvironment(BaseEnvironment):
    """
    2D physics world with Newtonian mechanics.
    
    Gravity pulls things down, balls bounce off walls
    and each other, agent can apply forces.
    """

    def __init__(
        self,
        num_balls: int = 4,
        num_obstacles: int = 2,
        world_size: float = 10.0,
        gravity: float = -9.8,
        dt: float = 0.02,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        self.num_balls = num_balls
        self.num_obstacles = num_obstacles
        self.world_size = world_size
        self.gravity = gravity
        self.dt = dt
        self.max_steps = max_steps
        
        self.rng = np.random.RandomState(seed)
        self.balls = []
        self.steps = 0
        
        # Agent is the first ball (color=0)
        # Obstacles are static (color=1, fixed)
        # Other balls are dynamic (color=2)
        self._obs_dim = num_balls * 6  # x, y, vx, vy, radius, color for each
        self._action_dim = 2  # force x, force y on agent
    
    def get_observation_dim(self) -> int:
        return self._obs_dim
    
    def get_action_dim(self) -> int:
        return self._action_dim
    
    def reset(self) -> np.ndarray:
        """Reset with random ball positions."""
        self.steps = 0
        self.balls = []
        
        # Agent ball (red, controllable)
        agent = Ball(
            x=self.rng.uniform(1, self.world_size - 1),
            y=self.rng.uniform(5, self.world_size - 1),
            vx=0, vy=0,
            radius=0.5, mass=1.0,
            color=0, fixed=False,
        )
        self.balls.append(agent)
        
        # Obstacle balls (green, fixed)
        for _ in range(self.num_obstacles):
            obs = Ball(
                x=self.rng.uniform(1, self.world_size - 1),
                y=self.rng.uniform(1, 4),
                radius=0.8, mass=float('inf'),
                color=1, fixed=True,
            )
            self.balls.append(obs)
        
        # Dynamic balls (blue)
        for _ in range(self.num_balls - 1 - self.num_obstacles):
            ball = Ball(
                x=self.rng.uniform(1, self.world_size - 1),
                y=self.rng.uniform(1, self.world_size - 1),
                vx=self.rng.uniform(-2, 2),
                vy=self.rng.uniform(-2, 2),
                radius=0.4, mass=1.0,
                color=2, fixed=False,
            )
            self.balls.append(ball)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply force and simulate physics."""
        self.steps += 1
        
        # Apply force to agent ball
        force = np.clip(action, -50, 50).astype(np.float64)
        agent = self.balls[0]
        if not agent.fixed:
            agent.vel += (force / agent.mass) * self.dt
        
        # Apply gravity
        for ball in self.balls:
            if not ball.fixed:
                ball.vel[1] += self.gravity * self.dt
        
        # Update positions
        for ball in self.balls:
            if not ball.fixed:
                ball.pos += ball.vel * self.dt
        
        # Wall collisions
        for ball in self.balls:
            if ball.fixed:
                continue
            for dim in range(2):
                if ball.pos[dim] - ball.radius < 0:
                    ball.pos[dim] = ball.radius
                    ball.vel[dim] *= -0.8  # bounce with energy loss
                elif ball.pos[dim] + ball.radius > self.world_size:
                    ball.pos[dim] = self.world_size - ball.radius
                    ball.vel[dim] *= -0.8
        
        # Ball-ball collisions
        collisions = 0
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                if self._collide(self.balls[i], self.balls[j]):
                    self._resolve_collision(self.balls[i], self.balls[j])
                    collisions += 1
        
        # Compute reward based on agent position relative to other balls
        reward = 0.0
        
        # Small energy penalty (encourages efficient movement)
        reward -= 0.001 * np.sum(agent.vel ** 2)
        
        # Penalty for going out of bounds (shouldn't happen but just in case)
        if agent.pos[0] < 0 or agent.pos[0] > self.world_size:
            reward -= 1.0
        if agent.pos[1] < 0 or agent.pos[1] > self.world_size:
            reward -= 1.0
        
        done = self.steps >= self.max_steps
        
        info = {
            "collisions": collisions,
            "agent_pos": agent.pos.copy(),
            "agent_vel": agent.vel.copy(),
        }
        
        return self._get_observation(), reward, done, info
    
    def _collide(self, ball1: Ball, ball2: Ball) -> bool:
        """Check if two balls are overlapping."""
        dist = np.linalg.norm(ball1.pos - ball2.pos)
        return dist < (ball1.radius + ball2.radius)
    
    def _resolve_collision(self, ball1: Ball, ball2: Ball):
        """Resolve elastic collision between two balls."""
        # Separation vector
        diff = ball1.pos - ball2.pos
        dist = np.linalg.norm(diff)
        
        if dist < 1e-10:
            return
        
        normal = diff / dist
        
        # Separate balls
        overlap = (ball1.radius + ball2.radius) - dist
        if overlap > 0:
            if ball1.fixed:
                ball2.pos -= normal * overlap
            elif ball2.fixed:
                ball1.pos += normal * overlap
            else:
                ball1.pos += normal * overlap * 0.5
                ball2.pos -= normal * overlap * 0.5
        
        # Compute relative velocity
        rel_vel = ball1.vel - ball2.vel
        vel_along_normal = np.dot(rel_vel, normal)
        
        # Don't resolve if balls are moving apart
        if vel_along_normal > 0:
            return
        
        # Elastic collision with restitution
        restitution = 0.9
        
        if ball1.fixed:
            ball2.vel -= (1 + restitution) * vel_along_normal * normal
        elif ball2.fixed:
            ball1.vel += (1 + restitution) * vel_along_normal * normal
        else:
            # Standard elastic collision formula
            m1, m2 = ball1.mass, ball2.mass
            impulse = -(1 + restitution) * vel_along_normal / (1/m1 + 1/m2)
            
            ball1.vel += (impulse / m1) * normal
            ball2.vel -= (impulse / m2) * normal
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []
        
        # Normalize positions and velocities
        for ball in self.balls:
            obs.append(ball.pos[0] / self.world_size)    # x normalized
            obs.append(ball.pos[1] / self.world_size)    # y normalized
            obs.append(ball.vel[0] / 10.0)               # vx normalized
            obs.append(ball.vel[1] / 10.0)               # vy normalized
            obs.append(ball.radius)                       # radius
            obs.append(float(ball.color) / 2.0)          # color normalized
        
        return np.array(obs, dtype=np.float32)
    
    def render(self) -> np.ndarray:
        """Render physics world as RGB image."""
        resolution = 100
        img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 30  # dark background
        
        colors = {
            0: (255, 100, 50),    # agent: orange-red
            1: (50, 200, 50),     # obstacles: green
            2: (50, 100, 255),    # dynamic: blue
        }
        
        for ball in self.balls:
            # Convert world coords to pixel coords
            px = int(np.clip(ball.pos[0] / self.world_size * (resolution - 1), 0, resolution - 1))
            py = int(np.clip((1 - ball.pos[1] / self.world_size) * (resolution - 1), 0, resolution - 1))
            pr = max(1, int(ball.radius / self.world_size * resolution))
            
            # Draw ball as filled circle (simplified)
            y_lo = max(0, py - pr)
            y_hi = min(resolution, py + pr + 1)
            x_lo = max(0, px - pr)
            x_hi = min(resolution, px + pr + 1)
            
            color = colors.get(ball.color, (200, 200, 200))
            img[y_lo:y_hi, x_lo:x_hi] = color
        
        return img
    
    def get_state(self) -> dict:
        """Full state for debugging."""
        return {
            "balls": [
                {
                    "pos": b.pos.copy(),
                    "vel": b.vel.copy(),
                    "radius": b.radius,
                    "color": b.color,
                    "fixed": b.fixed,
                }
                for b in self.balls
            ],
            "steps": self.steps,
        }
