"""Spatial hashing for efficient collision detection.

This module implements a spatial hashing system for efficient collision detection
between agents in the simulation. It uses a uniform grid to partition the world
space and only checks collisions between agents in the same or adjacent bins.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from .agent import Agent
from ..config import SimulationConfig
from dataclasses import dataclass

@dataclass
class Bin:
    """A bin in the spatial hash grid."""
    agents: Set[Agent]
    bin_index: Tuple[int, int]

class SpatialHash:
    """Spatial hashing system for efficient collision detection.
    
    This class implements a spatial hashing system that partitions the world
    into a uniform grid of bins. Each bin contains a set of agents, and
    collision detection only needs to check agents in the same or adjacent bins.
    
    Attributes:
        config: Simulation configuration containing world size and agent radius
        bin_size: Size of each bin in the grid (based on agent diameter)
        num_bins_x: Number of bins in the x direction
        num_bins_y: Number of bins in the y direction
        bins: Dictionary mapping bin indices to Bin objects
        agent_bins: Dictionary mapping agents to their current bin indices
        collision_checks: Counter for number of collision checks performed
        
    Methods:
        insert(agent): Insert an agent into the appropriate bin
        remove(agent): Remove an agent from its bin
        update(agent): Update an agent's position in the spatial hash
        get_potential_collisions(agent): Get agents that could collide with the given agent
        check_collision(agent1, agent2): Check if two agents are colliding
        clear(): Remove all agents from the spatial hash
        remove_agent(agent): Remove an agent from the spatial hash
        cleanup_inactive_agents(): Remove all inactive agents from the spatial hash
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize the spatial hash system.
        
        Args:
            config: Simulation configuration containing world size and agent radius
        """
        self.config = config
        self.bin_size = config.AGENT_RADIUS * 2  # Bin size based on agent diameter
        self.num_bins_x = int(np.ceil(config.WORLD_SIZE[0] / self.bin_size))
        self.num_bins_y = int(np.ceil(config.WORLD_SIZE[1] / self.bin_size))
        self.bins: Dict[Tuple[int, int], Bin] = {}
        self.agent_bins: Dict[Agent, Tuple[int, int]] = {}
        self.collision_checks = 0
        
        # Initialize all bins
        self._initialize_bins()
    
    def _initialize_bins(self) -> None:
        """Initialize all bins in the grid."""
        for x in range(self.num_bins_x):
            for y in range(self.num_bins_y):
                self.bins[(x, y)] = Bin(set(), (x, y))
    
    def _get_bin_index(self, position: np.ndarray) -> Tuple[int, int]:
        """Get the bin index for a given position.
        
        Args:
            position: Position vector [x, y]
            
        Returns:
            Tuple of (bin_x, bin_y) indices
        """
        bin_x = int(position[0] / self.bin_size)
        bin_y = int(position[1] / self.bin_size)
        return (bin_x, bin_y)
    
    def insert(self, agent: Agent) -> None:
        """Insert an agent into the spatial hash.
        
        Args:
            agent: Agent to insert
        """
        bin_index = self._get_bin_index(agent.position)
        self.bins[bin_index].agents.add(agent)
        self.agent_bins[agent] = bin_index
    
    def remove(self, agent: Agent) -> None:
        """Remove an agent from the spatial hash.
        
        Args:
            agent: Agent to remove
        """
        if agent in self.agent_bins:
            bin_index = self.agent_bins[agent]
            self.bins[bin_index].agents.discard(agent)
            del self.agent_bins[agent]
    
    def update(self, agent: Agent) -> None:
        """Update an agent's position in the spatial hash.
        
        Args:
            agent: Agent to update
        """
        self.remove(agent)
        self.insert(agent)
    
    def get_potential_collisions(self, agent: Agent) -> Set[Agent]:
        """Get set of agents that could potentially collide with the given agent.
        
        Args:
            agent: Agent to check collisions for
            
        Returns:
            Set of agents that could potentially collide
        """
        potential_collisions = set()
        bin_index = self._get_bin_index(agent.position)
        
        # Check current bin and adjacent bins
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x = bin_index[0] + dx
                check_y = bin_index[1] + dy
                
                # Handle world boundaries
                if (0 <= check_x < self.num_bins_x and 
                    0 <= check_y < self.num_bins_y):
                    check_bin = self.bins[(check_x, check_y)]
                    # Only include active agents
                    potential_collisions.update(a for a in check_bin.agents if a.is_active)
        
        # Remove the agent itself from potential collisions
        potential_collisions.discard(agent)
        self.collision_checks += len(potential_collisions)
        
        return potential_collisions
    
    def check_collision(self, agent1: Agent, agent2: Agent) -> bool:
        """Check if two agents are colliding.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            True if agents are colliding, False otherwise
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
        self.agent_bins.clear()
        self._initialize_bins()  # Reinitialize empty bins
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the spatial hash.
        
        Args:
            agent (Agent): Agent to remove
        """
        if agent in self.agent_bins:
            bin_idx = self.agent_bins[agent]
            if bin_idx in self.bins:
                self.bins[bin_idx].agents.discard(agent)
            del self.agent_bins[agent]
    
    def cleanup_inactive_agents(self) -> None:
        """Remove all inactive agents from the spatial hash."""
        inactive_agents = [agent for agent in self.agent_bins.keys() if not agent.is_active]
        for agent in inactive_agents:
            self.remove(agent) 