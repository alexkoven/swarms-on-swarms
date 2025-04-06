"""Agent representation for the swarm simulation environment."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from ..config import config

@dataclass
class Agent:
    """Represents an individual agent in the swarm simulation.
    
    Attributes:
        position (np.ndarray): 2D position vector [x, y]
        velocity (np.ndarray): 2D velocity vector [vx, vy]
        team_id (int): Team identifier (0-based)
        radius (float): Agent's collision radius
        is_active (bool): Whether the agent is currently active in the simulation
    """
    
    position: np.ndarray
    velocity: np.ndarray
    team_id: int
    radius: float = config.AGENT_RADIUS
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate and initialize agent attributes after creation."""
        # Ensure position and velocity are numpy arrays
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        
        # Validate dimensions
        assert self.position.shape == (2,), "Position must be a 2D vector"
        assert self.velocity.shape == (2,), "Velocity must be a 2D vector"
        
        # Validate team_id
        assert 0 <= self.team_id < config.NUM_TEAMS, f"Team ID must be between 0 and {config.NUM_TEAMS - 1}"
        
        # Validate radius
        assert self.radius > 0, "Agent radius must be positive"
        
        # Clip velocity to max speed
        speed = np.linalg.norm(self.velocity)
        if speed > config.MAX_VELOCITY:
            self.velocity = self.velocity / speed * config.MAX_VELOCITY
    
    def update_position(self, dt: float) -> None:
        """Update agent's position based on current velocity and time step.
        
        Args:
            dt (float): Time step for position update
        """
        assert dt > 0, "Time step must be positive"
        self.position += self.velocity * dt
        
        # Handle world boundaries (wrap around)
        self.position = self.position % config.WORLD_SIZE
    
    def set_velocity(self, new_velocity: np.ndarray) -> None:
        """Set agent's velocity with speed limit enforcement.
        
        Args:
            new_velocity (np.ndarray): New velocity vector [vx, vy]
        """
        new_velocity = np.asarray(new_velocity, dtype=np.float64)
        assert new_velocity.shape == (2,), "Velocity must be a 2D vector"
        
        # Clip velocity to max speed
        speed = np.linalg.norm(new_velocity)
        if speed > config.MAX_VELOCITY:
            self.velocity = new_velocity / speed * config.MAX_VELOCITY
        else:
            self.velocity = new_velocity
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get the current state of the agent.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: (position, velocity, team_id)
        """
        return self.position.copy(), self.velocity.copy(), self.team_id
    
    def deactivate(self) -> None:
        """Deactivate the agent from the simulation."""
        self.is_active = False
    
    def __eq__(self, other: object) -> bool:
        """Compare two agents for equality.
        
        Args:
            other (object): Object to compare with
            
        Returns:
            bool: True if agents are equal, False otherwise
        """
        if not isinstance(other, Agent):
            return NotImplemented
        return (np.array_equal(self.position, other.position) and
                np.array_equal(self.velocity, other.velocity) and
                self.team_id == other.team_id and
                self.radius == other.radius and
                self.is_active == other.is_active)
    
    def __hash__(self) -> int:
        """Compute hash value for the agent.
        
        Returns:
            int: Hash value
        """
        # Convert position and velocity to tuples for hashing
        pos_tuple = tuple(self.position)
        vel_tuple = tuple(self.velocity)
        return hash((pos_tuple, vel_tuple, self.team_id, self.radius, self.is_active))
    
    def check_collision(self, other: 'Agent') -> bool:
        """Check if this agent collides with another agent.
        
        Args:
            other: Other agent to check collision with
            
        Returns:
            True if agents collide, False otherwise
        """
        # Calculate distance between agents
        diff = self.position - other.position
        distance = np.linalg.norm(diff)
        
        # Collision occurs if distance is less than sum of radii
        return distance < (self.radius + other.radius) 