"""Main simulation environment managing agent states and simulation loop."""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from .agent import Agent
from .spatial_hash import SpatialHash
from ..config import config

class SwarmEnv:
    """Main simulation environment for swarm-vs-swarm interactions.
    
    This class manages the simulation state, including agent creation,
    updates, collision detection, and state observations.
    
    Attributes:
        agents (List[Agent]): List of all agents in the simulation
        spatial_hash (SpatialHash): Spatial partitioning for collision detection
        time_step (float): Current simulation time step
        step_count (int): Number of simulation steps executed
        team_counts (Dict[int, int]): Number of active agents per team
    """
    
    def __init__(self, time_step: float = config.TIME_STEP):
        """Initialize the simulation environment.
        
        Args:
            time_step (float): Time step for simulation updates
        """
        self.agents: List[Agent] = []
        self.spatial_hash = SpatialHash()
        self.time_step = time_step
        self.step_count = 0
        self.team_counts = {i: 0 for i in range(config.NUM_TEAMS)}
        
        # Validate time step
        assert time_step > 0, "Time step must be positive"
        
        # Initialize agents for each team
        self.initialize_agents()
    
    def initialize_agents(self) -> None:
        """Initialize agents for each team with random positions and velocities."""
        margin = 100.0  # Keep agents away from world boundaries
        world_size = config.WORLD_SIZE - 2 * margin
        
        for team_id in range(config.NUM_TEAMS):
            # Determine team's starting area
            if team_id == 0:
                # Team 0 starts on the left side
                x_range = (margin, world_size / 2)
            else:
                # Team 1 starts on the right side
                x_range = (world_size / 2 + margin, world_size)
            
            # Create agents for this team
            for _ in range(config.AGENTS_PER_TEAM):
                # Random position within team's area
                position = np.array([
                    np.random.uniform(*x_range),
                    np.random.uniform(margin, world_size)
                ])
                
                # Random initial velocity
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0, config.MAX_VELOCITY)
                velocity = speed * np.array([np.cos(angle), np.sin(angle)])
                
                self.add_agent(position, velocity, team_id)
    
    def add_agent(self, position: np.ndarray, velocity: np.ndarray, team_id: int) -> Agent:
        """Add a new agent to the simulation.
        
        Args:
            position (np.ndarray): Initial position vector [x, y]
            velocity (np.ndarray): Initial velocity vector [vx, vy]
            team_id (int): Team identifier
            
        Returns:
            Agent: The newly created agent
        """
        agent = Agent(position, velocity, team_id)
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
            agent.deactivate()
            self.spatial_hash.remove(agent)
            self.team_counts[agent.team_id] -= 1
            self.agents.remove(agent)
    
    def get_team_agents(self, team_id: int) -> List[Agent]:
        """Get all active agents for a specific team.
        
        Args:
            team_id (int): Team identifier
            
        Returns:
            List[Agent]: List of active agents in the team
        """
        return [agent for agent in self.agents if agent.team_id == team_id and agent.is_active]
    
    def get_nearby_agents(self, agent: Agent) -> Set[Agent]:
        """Get all agents near the specified agent.
        
        Args:
            agent (Agent): Agent to find neighbors for
            
        Returns:
            Set[Agent]: Set of nearby agents
        """
        return self.spatial_hash.get_potential_collisions(agent)
    
    def check_collisions(self, agent: Agent) -> List[Agent]:
        """Check for collisions with the specified agent.
        
        Args:
            agent (Agent): Agent to check collisions for
            
        Returns:
            List[Agent]: List of agents colliding with the specified agent
        """
        collisions = []
        potential_collisions = self.get_nearby_agents(agent)
        
        for other_agent in potential_collisions:
            if self.spatial_hash.check_collision(agent, other_agent):
                collisions.append(other_agent)
        
        return collisions
    
    def handle_collisions(self) -> List[Tuple[Agent, Agent]]:
        """Handle collisions between all agents.
        
        Returns:
            List[Tuple[Agent, Agent]]: List of agent pairs that collided
        """
        collisions = []
        checked_pairs = set()
        
        for agent in self.agents:
            if not agent.is_active:
                continue
                
            collision_agents = self.check_collisions(agent)
            for other_agent in collision_agents:
                # Create a unique pair identifier
                pair = tuple(sorted([id(agent), id(other_agent)]))
                if pair not in checked_pairs:
                    collisions.append((agent, other_agent))
                    checked_pairs.add(pair)
                    
                    # Handle collision by deactivating both agents
                    self.remove_agent(agent)
                    self.remove_agent(other_agent)
        
        return collisions
    
    def step(self) -> List[Tuple[Agent, Agent]]:
        """Execute one simulation step.
        
        Returns:
            List[Tuple[Agent, Agent]]: List of agent pairs that collided
        """
        # Update agent positions
        for agent in self.agents:
            if agent.is_active:
                old_position = agent.position.copy()
                agent.update_position(self.time_step)
                
                # Update spatial hash if position changed
                if not np.array_equal(old_position, agent.position):
                    self.spatial_hash.update(agent)
        
        # Handle collisions
        collisions = self.handle_collisions()
        
        # Update step count
        self.step_count += 1
        
        return collisions
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        self.agents.clear()
        self.spatial_hash.clear()
        self.step_count = 0
        self.team_counts = {i: 0 for i in range(config.NUM_TEAMS)}
    
    def get_state(self) -> Dict:
        """Get the current state of the simulation.
        
        Returns:
            Dict: Current simulation state including agent states and statistics
        """
        return {
            'step_count': self.step_count,
            'team_counts': self.team_counts.copy(),
            'agents': [
                {
                    'position': agent.position.copy(),
                    'velocity': agent.velocity.copy(),
                    'team_id': agent.team_id,
                    'is_active': agent.is_active
                }
                for agent in self.agents
            ]
        } 