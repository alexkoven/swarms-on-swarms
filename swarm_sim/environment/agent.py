"""Agent representation for the swarm simulation environment."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from ..config import SimulationConfig

@dataclass
class Agent:
    """Represents an individual agent in the swarm simulation.
    
    Attributes:
        position (np.ndarray): 2D position vector [x, y]
        velocity (np.ndarray): 2D velocity vector [vx, vy]
        team_id (int): Team identifier (0-based)
        config (SimulationConfig): Configuration for the simulation
        radius (float): Agent's collision radius
        is_active (bool): Whether the agent is currently active in the simulation
    """
    
    position: np.ndarray
    velocity: np.ndarray
    team_id: int
    config: SimulationConfig
    radius: float = None
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate and initialize agent attributes after creation."""
        # Ensure position and velocity are numpy arrays
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        
        # Set radius from config if not provided
        if self.radius is None:
            self.radius = self.config.AGENT_RADIUS
        
        # Validate dimensions
        assert self.position.shape == (2,), "Position must be a 2D vector"
        assert self.velocity.shape == (2,), "Velocity must be a 2D vector"
        
        # Validate team_id
        assert 0 <= self.team_id < self.config.NUM_TEAMS, f"Team ID must be between 0 and {self.config.NUM_TEAMS - 1}"
        
        # Validate radius
        assert self.radius > 0, "Agent radius must be positive"
        
        # Clip velocity to max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.config.MAX_VELOCITY:
            self.velocity = self.velocity / speed * self.config.MAX_VELOCITY
    
    def update_position(self, dt: float) -> None:
        """Update agent position based on velocity and boundary conditions."""
        new_position = self.position + self.velocity * dt

        if self.config.BOUNDARY_TYPE == "wrap":
            # Wrap around world boundaries
            new_position = np.mod(new_position, self.config.WORLD_SIZE)
        elif self.config.BOUNDARY_TYPE == "bounce":
            # Bounce off world boundaries
            for i in range(2):
                if new_position[i] < 0:
                    new_position[i] = 0
                    self.velocity[i] *= -1
                elif new_position[i] > self.config.WORLD_SIZE[i]:
                    new_position[i] = self.config.WORLD_SIZE[i]
                    self.velocity[i] *= -1

        self.position = new_position
    
    def set_velocity(self, velocity: np.ndarray) -> None:
        """Set the agent's velocity, enforcing speed limit."""
        velocity = np.asarray(velocity, dtype=np.float64)
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-10:  # Handle zero or near-zero velocity
            self.velocity = np.zeros(2, dtype=np.float64)
        elif speed > self.config.MAX_VELOCITY:
            # Scale velocity to maximum speed while preserving direction
            self.velocity = (velocity / speed) * self.config.MAX_VELOCITY
        else:
            self.velocity = velocity.copy()
    
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
        # Only use immutable properties for hashing
        # We use id() to get a unique identifier for this agent instance
        # This ensures the hash doesn't change when position/velocity change
        return hash((id(self), self.team_id))
    
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