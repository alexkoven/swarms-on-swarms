"""Tests for the main simulation runner."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from ..main import SwarmSimulation
from ..environment.swarm_env import SwarmEnv
from ..config import SimulationConfig

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig.default()

@pytest.fixture
def env(test_config):
    """Create a test environment."""
    return SwarmEnv(test_config)

@pytest.fixture
def simulation(env):
    """Create a test simulation."""
    return SwarmSimulation(env)

def test_initialization(simulation, test_config):
    """Test simulation initialization."""
    # Check controller initialization
    assert len(simulation.controllers) == test_config.NUM_TEAMS
    for team_id in range(test_config.NUM_TEAMS):
        assert team_id in simulation.controllers
    
    # Check visualization setup if enabled
    if test_config.VISUALIZE:
        assert hasattr(simulation, 'fig')
        assert hasattr(simulation, 'ax')
        assert len(simulation.scatter_plots) == test_config.NUM_TEAMS

def test_update_visualization(simulation, test_config):
    """Test visualization update."""
    if test_config.VISUALIZE:
        # Run one update step
        artists = simulation.update_visualization(0)

        # Check that artists were returned (one per team plus status text)
        assert len(artists) == test_config.NUM_TEAMS + 1

        # Check that all artists are valid matplotlib objects
        for artist in artists:
            assert isinstance(artist, (plt.Artist, plt.Text))

def test_run_without_visualization(simulation, test_config, monkeypatch):
    """Test running simulation without visualization."""
    # Disable visualization
    monkeypatch.setattr(test_config, 'VISUALIZE', False)
    
    # Run for a few steps
    num_steps = 10
    simulation.run(num_steps)
    
    # Check that agents were updated
    for team_id in range(test_config.NUM_TEAMS):
        team_agents = simulation.env.get_team_agents(team_id)
        for agent in team_agents:
            if agent.is_active:
                # Agents should have non-zero velocities due to random control
                assert not np.array_equal(agent.velocity, np.zeros(2))

def test_run_with_steps(simulation, test_config, monkeypatch):
    """Test running simulation for specific number of steps."""
    # Disable visualization for testing
    monkeypatch.setattr(test_config, 'VISUALIZE', False)
    
    # Run for specific number of steps
    num_steps = 5
    simulation.run(num_steps)
    
    # Verify simulation ran
    assert simulation.env.step_count == num_steps 