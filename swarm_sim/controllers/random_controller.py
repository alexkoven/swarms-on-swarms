"""Random controller for basic agent movement control."""

from typing import Dict, List, Tuple
import numpy as np
from ..environment.agent import Agent
from ..config import config

class RandomController:
    """Controller that generates random actions for agents.
    
    This controller provides basic movement control by generating random
    velocity changes within specified constraints. It is primarily used for
    testing and demonstration purposes, as it does not implement any
    strategic behavior.
    
    The controller generates random velocity changes in both x and y
    directions, which are then applied to the agent's current velocity.
    The changes are bounded by max_velocity_change to ensure smooth
    movement and prevent sudden direction changes.
    
    Attributes:
        max_velocity_change (float): Maximum allowed change in velocity per step
        action_space (Tuple[float, float]): Range for random action values,
            defined as (-max_velocity_change, max_velocity_change)
    
    Example:
        >>> controller = RandomController(max_velocity_change=1.0)
        >>> action = controller.get_action(agent)
        >>> agent.set_velocity(agent.velocity + action)
    """
    
    def __init__(self, max_velocity_change: float = config.MAX_VELOCITY_CHANGE):
        """Initialize the random controller.
        
        Args:
            max_velocity_change (float): Maximum allowed change in velocity per step
        """
        self.max_velocity_change = max_velocity_change
        self.action_space = (-max_velocity_change, max_velocity_change)
    
    def get_action(self, agent: Agent) -> np.ndarray:
        """Generate a random action for the agent.
        
        Args:
            agent (Agent): Agent to generate action for
            
        Returns:
            np.ndarray: Random velocity change vector [dvx, dvy]
        """
        # Generate random velocity changes within constraints
        dvx = np.random.uniform(*self.action_space)
        dvy = np.random.uniform(*self.action_space)
        
        return np.array([dvx, dvy])
    
    def apply_action(self, agent: Agent, action: np.ndarray) -> None:
        """Apply the action to the agent's velocity.
        
        Args:
            agent (Agent): Agent to apply action to
            action (np.ndarray): Velocity change vector [dvx, dvy]
        """
        # Update velocity with action
        new_velocity = agent.velocity + action
        
        # Ensure velocity magnitude doesn't exceed maximum
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > config.MAX_VELOCITY:
            new_velocity = new_velocity * (config.MAX_VELOCITY / velocity_magnitude)
        
        agent.velocity = new_velocity
    
    def control_team(self, agents: List[Agent]) -> None:
        """Control all agents in a team.
        
        Args:
            agents (List[Agent]): List of agents to control
        """
        for agent in agents:
            if agent.is_active:
                action = self.get_action(agent)
                self.apply_action(agent, action) 