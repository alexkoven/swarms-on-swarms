"""Configuration settings for the swarm simulation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    
    # World settings
    WORLD_SIZE: float = float(os.getenv("WORLD_SIZE", "1000.0"))
    TIME_STEP: float = float(os.getenv("TIME_STEP", "0.1"))
    
    # Agent settings
    AGENT_RADIUS: float = float(os.getenv("AGENT_RADIUS", "5.0"))
    MAX_VELOCITY: float = float(os.getenv("MAX_VELOCITY", "50.0"))
    MAX_VELOCITY_CHANGE: float = float(os.getenv("MAX_VELOCITY_CHANGE", "10.0"))
    
    # Team settings
    NUM_TEAMS: int = int(os.getenv("NUM_TEAMS", "2"))
    AGENTS_PER_TEAM: int = int(os.getenv("AGENTS_PER_TEAM", "50"))  # Reduced for better visualization
    
    # Spatial hash settings
    BIN_SIZE: float = float(os.getenv("SPATIAL_HASH_BIN_SIZE", "10.0"))
    
    # Visualization settings
    VISUALIZE: bool = os.getenv("VISUALIZE", "true").lower() == "true"
    FRAME_RATE: int = int(os.getenv("FRAME_RATE", "30"))
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "1000"))
    
    # Team colors
    TEAM_COLORS: List[str] = None
    
    def __post_init__(self):
        """Initialize derived attributes."""
        if self.TEAM_COLORS is None:
            self.TEAM_COLORS = ['red', 'blue', 'green', 'yellow']

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.WORLD_SIZE > 0, "World size must be positive"
        assert self.NUM_TEAMS > 0, "Number of teams must be positive"
        assert self.AGENT_RADIUS > 0, "Agent radius must be positive"
        assert self.MAX_VELOCITY > 0, "Maximum velocity must be positive"
        assert self.TIME_STEP > 0, "Time step must be positive"
        assert self.BOUNDARY_TYPE in ["wrap", "bounce"], "Boundary type must be 'wrap' or 'bounce'"

# Create global config instance
config = SimulationConfig() 