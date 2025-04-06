"""Main simulation environment for swarm-vs-swarm simulation.

This module implements the main simulation environment that manages agent states
and the simulation loop. It handles:
- Agent creation and management
- State updates
- Collision detection and handling
- Environment observation
"""

import time
import numpy as np
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass

from .agent import Agent
from .spatial_hash import SpatialHash
from ..config import SimulationConfig

@dataclass
class SimulationState:
    """Current state of the simulation.
    
    Attributes:
        step_count: Current simulation step
        active_agents: List of active agents
        team_counts: Dictionary mapping team IDs to number of active agents
        step_time: Time taken for last step in seconds
    """
    step_count: int
    active_agents: List[Agent]
    team_counts: Dict[int, int]
    step_time: float

class SwarmEnv:
    """Main simulation environment for swarm-vs-swarm simulation.
    
    This class manages the simulation state and provides methods for:
    - Adding and removing agents
    - Updating agent states
    - Detecting and handling collisions
    - Getting environment observations
    
    Attributes:
        config: Simulation configuration
        agents: List of all agents in the simulation
        spatial_hash: Spatial hash for efficient collision detection
        time_step: Simulation time step
        step_count: Current simulation step
        team_counts: Dictionary mapping team IDs to number of active agents
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize the simulation environment.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.agents: List[Agent] = []
        self.spatial_hash = SpatialHash(config)
        self.time_step = config.TIME_STEP
        self.step_count = 0
        self.team_counts: Dict[int, int] = {}
        
        # Initialize team counts
        for team_id in range(config.NUM_TEAMS):
            self.team_counts[team_id] = 0
        
        # Initialize agents
        self.initialize_agents()
    
    def initialize_agents(self) -> None:
        """Initialize agents for each team with random positions and velocities."""
        margin = self.config.AGENT_RADIUS * 2  # Keep agents away from boundaries
        world_size = np.array(self.config.WORLD_SIZE)
        
        for team_id in range(self.config.NUM_TEAMS):
            # Calculate team's starting area
            team_width = (world_size[0] - 2 * margin) / self.config.NUM_TEAMS
            x_start = margin + team_id * team_width
            x_end = x_start + team_width
            
            # Create agents for this team
            for _ in range(self.config.AGENTS_PER_TEAM):
                # Random position within team's area
                position = np.array([
                    np.random.uniform(x_start, x_end),
                    np.random.uniform(margin, world_size[1] - margin)
                ])
                
                # Random velocity
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0, self.config.MAX_VELOCITY)
                velocity = np.array([
                    speed * np.cos(angle),
                    speed * np.sin(angle)
                ])
                
                self.add_agent(position, velocity, team_id)
    
    def add_agent(self, position: np.ndarray, velocity: np.ndarray, team_id: int) -> Agent:
        """Add a new agent to the simulation.
        
        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy]
            team_id: Team ID for the agent
            
        Returns:
            The created agent
        """
        agent = Agent(position, velocity, team_id)
        self.agents.append(agent)
        self.spatial_hash.insert(agent)
        self.team_counts[team_id] += 1
        return agent
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the simulation.
        
        Args:
            agent: Agent to remove
        """
        if agent in self.agents:
            self.agents.remove(agent)
            self.spatial_hash.remove(agent)
            self.team_counts[agent.team_id] -= 1
    
    def get_team_agents(self, team_id: int) -> List[Agent]:
        """Get all active agents for a specific team.
        
        Args:
            team_id: Team ID to get agents for
            
        Returns:
            List of active agents for the team
        """
        return [a for a in self.agents if a.team_id == team_id and a.is_active]
    
    def get_nearby_agents(self, agent: Agent) -> Set[Agent]:
        """Get all agents that could potentially collide with the given agent.
        
        Args:
            agent: Agent to check for nearby agents
            
        Returns:
            Set of nearby agents
        """
        return self.spatial_hash.get_potential_collisions(agent)
    
    def handle_collisions(self) -> None:
        """Handle collisions between agents.
        
        When agents from different teams collide, they are deactivated.
        """
        for agent in self.agents:
            if not agent.is_active:
                continue
                
            nearby_agents = self.get_nearby_agents(agent)
            for other in nearby_agents:
                if not other.is_active or agent.team_id == other.team_id:
                    continue
                    
                if agent.check_collision(other):
                    agent.is_active = False
                    other.is_active = False
                    break
    
    def step(self) -> SimulationState:
        """Execute one simulation step.
        
        This method:
        1. Updates agent positions and velocities
        2. Handles collisions
        3. Updates spatial hash
        4. Returns current simulation state
        
        Returns:
            Current simulation state
        """
        start_time = time.time()
        
        # Update agent states
        for agent in self.agents:
            if not agent.is_active:
                continue
                
            # Update position
            agent.position += agent.velocity * self.time_step
            
            # Handle world boundaries
            agent.position[0] %= self.config.WORLD_SIZE[0]
            agent.position[1] %= self.config.WORLD_SIZE[1]
            
            # Update spatial hash
            self.spatial_hash.update(agent)
        
        # Handle collisions
        self.handle_collisions()
        
        # Update step count
        self.step_count += 1
        
        # Calculate step time
        step_time = time.time() - start_time
        
        # Return current state
        return SimulationState(
            step_count=self.step_count,
            active_agents=[a for a in self.agents if a.is_active],
            team_counts=self.team_counts.copy(),
            step_time=step_time
        )
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        self.agents.clear()
        self.spatial_hash = SpatialHash(self.config)
        self.step_count = 0
        self.team_counts = {i: 0 for i in range(self.config.NUM_TEAMS)}
        self.initialize_agents()
    
    def get_state(self) -> SimulationState:
        """Get the current state of the simulation.
        
        Returns:
            Current simulation state
        """
        return SimulationState(
            step_count=self.step_count,
            active_agents=[a for a in self.agents if a.is_active],
            team_counts=self.team_counts.copy(),
            step_time=0.0  # Not applicable for current state
        ) 