"""Random controller for swarm agents.

This module implements a simple random controller that moves agents in random directions.
It's used for testing and demonstration purposes.
"""

import numpy as np
from typing import List
from ..environment.agent import Agent

class RandomController:
    """Random controller for swarm agents.
    
    This controller moves agents in random directions while respecting:
    - Maximum velocity limits
    - Maximum velocity change per step
    - World boundaries
    
    Attributes:
        max_velocity_change: Maximum change in velocity per step
    """
    
    def __init__(self, max_velocity_change: float = None):
        """Initialize the controller.
        
        Args:
            max_velocity_change: Maximum change in velocity per step.
                               If None, will be set from agent config.
        """
        self.max_velocity_change = max_velocity_change
    
    def get_action(self, agent: Agent) -> np.ndarray:
        """Get a random action for the agent.
        
        Args:
            agent: Agent to control
            
        Returns:
            New velocity vector
        """
        # Use agent's config if not specified
        max_change = (self.max_velocity_change 
                     if self.max_velocity_change is not None 
                     else agent.config.MAX_VELOCITY_CHANGE)
        
        # Random change in velocity
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, max_change)
        change = np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle)
        ])
        
        # Apply change to current velocity
        new_velocity = agent.velocity + change
        
        # Limit speed
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > agent.config.MAX_VELOCITY:
            new_velocity = new_velocity * (agent.config.MAX_VELOCITY / velocity_magnitude)
        
        return new_velocity
    
    def apply_action(self, agent: Agent, action: np.ndarray) -> None:
        """Apply the action to the agent.
        
        Args:
            agent: Agent to control
            action: New velocity vector
        """
        agent.set_velocity(action)
    
    def control_team(self, agents: List[Agent]) -> None:
        """Control a team of agents.
        
        Args:
            agents: List of agents to control
        """
        for agent in agents:
            if not agent.is_active:
                continue
            action = self.get_action(agent)
            self.apply_action(agent, action) 