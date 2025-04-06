from dataclasses import dataclass, field
from typing import Dict, List
from .agent import Agent

@dataclass
class SimulationState:
    """Represents the state of the simulation at a point in time."""
    
    step_count: int
    team_counts: Dict[int, int]
    active_agents: List[Agent] = field(default_factory=list)  # Use active_agents instead of agents
    step_time: float = 0.0  # Time taken for the last simulation step

    def __post_init__(self):
        # Ensure that the team_counts dictionary is initialized
        if self.team_counts is None:
            self.team_counts = {}

    def add_agent(self, agent: Agent):
        """Adds a new agent to the simulation."""
        self.active_agents.append(agent)

    def remove_agent(self, agent: Agent):
        """Removes an agent from the simulation."""
        self.active_agents.remove(agent)

    def get_agents(self):
        """Returns the list of active agents in the simulation."""
        return self.active_agents

    def get_team_counts(self):
        """Returns the current team counts in the simulation."""
        return self.team_counts

    def get_step_count(self):
        """Returns the current step count in the simulation."""
        return self.step_count

    def update_step_count(self, new_step_count: int):
        """Updates the step count in the simulation."""
        self.step_count = new_step_count

    def update_team_counts(self, team_id: int, new_count: int):
        """Updates the team count in the simulation."""
        self.team_counts[team_id] = new_count

    def is_empty(self):
        """Returns True if there are no active agents in the simulation."""
        return len(self.active_agents) == 0

    def clear(self):
        """Clears all agents from the simulation."""
        self.active_agents.clear()
        self.team_counts.clear()
        self.step_count = 0

    def __str__(self):
        """Returns a string representation of the simulation state."""
        return f"Step: {self.step_count}, Team Counts: {self.team_counts}, Active Agents: {self.active_agents}" 