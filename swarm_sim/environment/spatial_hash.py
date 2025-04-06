"""Spatial hashing for efficient collision detection."""

from typing import Dict, List, Set, Tuple
import numpy as np
from .agent import Agent
from ..config import config

class SpatialHash:
    """Spatial hash grid for efficient collision detection.
    
    This class divides the world into uniform bins and tracks which agents are in each bin,
    allowing for efficient collision detection by only checking agents in nearby bins.
    
    Attributes:
        bin_size (float): Size of each bin
        bins (Dict[Tuple[int, int], Set[Agent]]): Map from bin indices to agents in that bin
        agent_bins (Dict[Agent, Tuple[int, int]]): Map from agents to their current bin
    """
    
    def __init__(self, bin_size: float = config.BIN_SIZE):
        """Initialize the spatial hash.
        
        Args:
            bin_size (float): Size of each bin
        """
        self.bin_size = bin_size
        self.bins: Dict[Tuple[int, int], Set[Agent]] = {}
        self.agent_bins: Dict[Agent, Tuple[int, int]] = {}
    
    def get_bin_indices(self, position: np.ndarray) -> Tuple[int, int]:
        """Get the bin indices for a position.
        
        Args:
            position (np.ndarray): Position to get bin indices for
            
        Returns:
            Tuple[int, int]: Bin indices (x, y)
        """
        return (
            int(position[0] / self.bin_size),
            int(position[1] / self.bin_size)
        )
    
    def insert(self, agent: Agent) -> None:
        """Insert an agent into the appropriate bin.
        
        Args:
            agent (Agent): Agent to insert
        """
        bin_indices = self.get_bin_indices(agent.position)
        if bin_indices not in self.bins:
            self.bins[bin_indices] = set()
        self.bins[bin_indices].add(agent)
        self.agent_bins[agent] = bin_indices
    
    def remove(self, agent: Agent) -> None:
        """Remove an agent from its bin.
        
        Args:
            agent (Agent): Agent to remove
        """
        if agent in self.agent_bins:
            bin_indices = self.agent_bins[agent]
            if bin_indices in self.bins:
                self.bins[bin_indices].discard(agent)
                if not self.bins[bin_indices]:
                    del self.bins[bin_indices]
            del self.agent_bins[agent]
    
    def update(self, agent: Agent) -> None:
        """Update an agent's position in the spatial hash.
        
        Args:
            agent (Agent): Agent to update
        """
        self.remove(agent)
        self.insert(agent)
    
    def get_potential_collisions(self, agent: Agent) -> List[Agent]:
        """Get all agents that could potentially collide with the given agent.
        
        Args:
            agent (Agent): Agent to check collisions for
            
        Returns:
            List[Agent]: List of agents that could potentially collide
        """
        bin_indices = self.get_bin_indices(agent.position)
        nearby_agents = []
        
        # Check current bin and adjacent bins
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_bin = (bin_indices[0] + dx, bin_indices[1] + dy)
                if check_bin in self.bins:
                    nearby_agents.extend(self.bins[check_bin])
        
        # Remove self from list
        if agent in nearby_agents:
            nearby_agents.remove(agent)
        
        return nearby_agents
    
    def check_collision(self, agent1: Agent, agent2: Agent) -> bool:
        """Check if two agents are colliding.
        
        Args:
            agent1 (Agent): First agent
            agent2 (Agent): Second agent
            
        Returns:
            bool: True if agents are colliding, False otherwise
        """
        if not agent1.is_active or not agent2.is_active:
            return False
            
        # Check if agents are on the same team
        if agent1.team_id == agent2.team_id:
            return False
            
        # Calculate distance between agents
        distance = np.linalg.norm(agent1.position - agent2.position)
        min_distance = agent1.radius + agent2.radius
        
        return distance < min_distance
    
    def clear(self) -> None:
        """Clear all agents from the spatial hash."""
        self.bins.clear()
        self.agent_bins.clear() 