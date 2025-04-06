"""Configuration for the swarm simulation.

This module contains all configuration parameters for the simulation, including:
- World parameters (size, time step)
- Agent parameters (radius, speed, etc.)
- Team parameters (number of teams, agents per team)
- Visualization parameters (colors, frame rate)
"""

from dataclasses import dataclass
from typing import Tuple
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file if it exists
load_dotenv()

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation.
    
    Attributes:
        WORLD_SIZE: Size of the world in (width, height)
        TIME_STEP: Simulation time step in seconds
        AGENT_RADIUS: Radius of each agent
        MAX_VELOCITY: Maximum agent velocity
        MAX_VELOCITY_CHANGE: Maximum change in velocity per step
        NUM_TEAMS: Number of teams in the simulation
        AGENTS_PER_TEAM: Number of agents per team
        TEAM_COLORS: Colors for each team in the visualization
        FRAME_RATE: Frame rate for visualization
        VISUALIZE: Whether to show visualization
    """
    
    # World parameters
    WORLD_SIZE: Tuple[float, float] = (1000.0, 1000.0)
    TIME_STEP: float = 0.016  # 60 FPS
    
    # Agent parameters
    AGENT_RADIUS: float = 5.0
    MAX_VELOCITY: float = 100.0
    MAX_VELOCITY_CHANGE: float = 10.0
    
    # Team parameters
    NUM_TEAMS: int = 2
    AGENTS_PER_TEAM: int = 1000  # Increased for visualization
    TEAM_COLORS: Tuple[str, ...] = ('red', 'blue')
    
    # Visualization parameters
    FRAME_RATE: int = 60
    VISUALIZE: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert len(self.WORLD_SIZE) == 2, "WORLD_SIZE must be a tuple of (width, height)"
        assert all(s > 0 for s in self.WORLD_SIZE), "World dimensions must be positive"
        assert self.TIME_STEP > 0, "TIME_STEP must be positive"
        assert self.AGENT_RADIUS > 0, "AGENT_RADIUS must be positive"
        assert self.MAX_VELOCITY > 0, "MAX_VELOCITY must be positive"
        assert self.MAX_VELOCITY_CHANGE > 0, "MAX_VELOCITY_CHANGE must be positive"
        assert self.NUM_TEAMS > 0, "NUM_TEAMS must be positive"
        assert self.AGENTS_PER_TEAM > 0, "AGENTS_PER_TEAM must be positive"
        assert len(self.TEAM_COLORS) >= self.NUM_TEAMS, "Not enough team colors"
        assert self.FRAME_RATE > 0, "FRAME_RATE must be positive"

# Create global config instance
config = SimulationConfig() 