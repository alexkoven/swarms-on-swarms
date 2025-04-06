"""Tests for the main simulation runner."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from ..main import SwarmSimulation
from ..environment.swarm_env import SwarmEnv
from ..config import config, SimulationConfig

@pytest.fixture
def env():
    """Create a test environment."""
    test_config = SimulationConfig()
    return SwarmEnv(test_config)

@pytest.fixture
def simulation(env):
    """Create a test simulation."""
    return SwarmSimulation(env)

def test_initialization(simulation):
    """Test simulation initialization."""
    # Check controller initialization
    assert len(simulation.controllers) == config.NUM_TEAMS
    for team_id in range(config.NUM_TEAMS):
        assert team_id in simulation.controllers
    
    # Check visualization setup if enabled
    if config.VISUALIZE:
        assert hasattr(simulation, 'fig')
        assert hasattr(simulation, 'ax')
        assert len(simulation.scatter_plots) == config.NUM_TEAMS

def test_update_visualization(simulation):
    """Test visualization update."""
    if config.VISUALIZE:
        # Run one update step
        artists = simulation.update_visualization(0)

        # Check that artists were returned (one per team plus status text)
        assert len(artists) == config.NUM_TEAMS + 1

        # Check that all artists are valid matplotlib objects
        for artist in artists:
            assert isinstance(artist, (plt.Artist, plt.Text))

def test_run_without_visualization(simulation, monkeypatch):
    """Test running simulation without visualization."""
    # Disable visualization
    monkeypatch.setattr(config, 'VISUALIZE', False)
    
    # Run for a few steps
    num_steps = 10
    simulation.run(num_steps)
    
    # Check that agents were updated
    for team_id in range(config.NUM_TEAMS):
        team_agents = simulation.env.get_team_agents(team_id)
        for agent in team_agents:
            if agent.is_active:
                # Agents should have non-zero velocities due to random control
                assert not np.array_equal(agent.velocity, np.zeros(2))

def test_run_with_steps(simulation, monkeypatch):
    """Test running simulation for specific number of steps."""
    # Disable visualization for testing
    monkeypatch.setattr(config, 'VISUALIZE', False)
    
    # Run for specific number of steps
    num_steps = 5
    simulation.run(num_steps)
    
    # Verify simulation ran
    assert simulation.env.step_count == num_steps 