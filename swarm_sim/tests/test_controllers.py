"""Tests for the random controller."""

import pytest
import numpy as np
from ..controllers.random_controller import RandomController
from ..environment.agent import Agent
from ..config import SimulationConfig

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig.default()

@pytest.fixture
def agent(test_config):
    """Create a test agent."""
    return Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)

def test_initialization(test_config):
    """Test controller initialization."""
    # Test with default max_velocity_change
    controller = RandomController()
    assert controller.max_velocity_change is None
    
    # Test with custom max_velocity_change
    max_change = 5.0
    controller = RandomController(max_velocity_change=max_change)
    assert controller.max_velocity_change == max_change

def test_get_action(test_config, agent):
    """Test action generation."""
    controller = RandomController()
    
    # Get action
    action = controller.get_action(agent)
    
    # Check action is a 2D vector
    assert action.shape == (2,)
    
    # Check action respects velocity limits
    speed = np.linalg.norm(action)
    assert speed <= agent.config.MAX_VELOCITY

def test_apply_action(test_config, agent):
    """Test action application."""
    controller = RandomController()
    
    # Set initial velocity
    initial_velocity = np.array([1.0, 1.0])
    agent.set_velocity(initial_velocity)
    
    # Apply action
    action = np.array([2.0, 2.0])
    controller.apply_action(agent, action)
    
    # Check velocity was updated
    assert np.array_equal(agent.velocity, action)

def test_control_team(test_config):
    """Test team control."""
    controller = RandomController()
    
    # Create team of agents
    team = [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 0, test_config)
    ]
    
    # Control team
    controller.control_team(team)
    
    # Check all agents were controlled
    for agent in team:
        assert np.linalg.norm(agent.velocity) <= agent.config.MAX_VELOCITY

def test_inactive_agents(test_config):
    """Test handling of inactive agents."""
    controller = RandomController()
    
    # Create team with mix of active and inactive agents
    team = [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 0, test_config)
    ]
    team[1].deactivate()  # Make middle agent inactive
    
    # Store initial velocities
    initial_velocities = [agent.velocity.copy() for agent in team]
    
    # Control team
    controller.control_team(team)
    
    # Check only active agents were controlled
    assert not np.array_equal(team[0].velocity, initial_velocities[0])
    assert np.array_equal(team[1].velocity, initial_velocities[1])
    assert not np.array_equal(team[2].velocity, initial_velocities[2]) 