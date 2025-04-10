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
        agent = Agent(position, velocity, team_id, self.config)
        self.agents.append(agent)
        self.spatial_hash.insert(agent)
        self.team_counts[team_id] += 1
        return agent
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the simulation.
        
        Args:
            agent (Agent): Agent to remove
        """
        if agent in self.agents:
            self.agents.remove(agent)
            self.team_counts[agent.team_id] -= 1
            agent.is_active = False
            self.spatial_hash.remove_agent(agent)
    
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
    
    def handle_collisions(self) -> List[Tuple[Agent, Agent]]:
        """Handle collisions between agents.
        
        Returns:
            List of tuples containing pairs of colliding agents
        """
        collisions = []
        processed_pairs = set()  # Track processed agent pairs to avoid double-counting
        
        # Create a randomized copy of the agents list to avoid processing bias
        agents_randomized = self.agents.copy()
        np.random.shuffle(agents_randomized)
        
        for agent in agents_randomized:
            if not agent.is_active:
                continue
                
            nearby = self.spatial_hash.get_potential_collisions(agent)
            for other in nearby:
                if not other.is_active or other is agent:
                    continue
                    
                # Only handle collisions between agents of different teams
                if agent.team_id == other.team_id:
                    continue
                    
                # Avoid processing the same pair twice
                pair = tuple(sorted([agent, other], key=lambda x: x.team_id))
                if pair in processed_pairs:
                    continue
                    
                # Check actual collision
                distance = np.linalg.norm(agent.position - other.position)
                if distance < 2 * agent.radius:
                    # Deactivate both agents
                    agent.is_active = False
                    other.is_active = False
                    self.team_counts[agent.team_id] -= 1
                    self.team_counts[other.team_id] -= 1
                    collisions.append((agent, other))
                    processed_pairs.add(pair)
                    
        return collisions
    
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
        
        # Clean up inactive agents from spatial hash first
        self.spatial_hash.cleanup_inactive_agents()
        
        # Update agent states
        for agent in self.agents:
            if not agent.is_active:
                continue
                
            # Update position
            new_position = agent.position + agent.velocity * self.time_step
            
            # Handle world boundaries with hard boundaries
            if new_position[0] - agent.radius < 0:
                new_position[0] = agent.radius
                agent.velocity[0] = -agent.velocity[0]
            elif new_position[0] + agent.radius > self.config.WORLD_SIZE[0]:
                new_position[0] = self.config.WORLD_SIZE[0] - agent.radius
                agent.velocity[0] = -agent.velocity[0]
                
            if new_position[1] - agent.radius < 0:
                new_position[1] = agent.radius
                agent.velocity[1] = -agent.velocity[1]
            elif new_position[1] + agent.radius > self.config.WORLD_SIZE[1]:
                new_position[1] = self.config.WORLD_SIZE[1] - agent.radius
                agent.velocity[1] = -agent.velocity[1]
            
            agent.position = new_position
            
            # Update spatial hash
            self.spatial_hash.update(agent)
        
        # Handle collisions
        self.handle_collisions()
        
        # Update step count
        self.step_count += 1
        
        # Calculate step time
        step_time = time.time() - start_time
        
        # Return current state with updated team counts
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
        """Get current simulation state.
        
        Returns:
            SimulationState object containing current state
        """
        return SimulationState(
            step_count=self.step_count,
            team_counts=self.team_counts.copy(),
            active_agents=[agent for agent in self.agents if agent.is_active],
            step_time=0.0  # We don't track step time in the environment
        ) 