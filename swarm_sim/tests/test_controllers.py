"""Tests for the RandomController class."""

import pytest
import numpy as np
from ..controllers.random_controller import RandomController
from ..environment.agent import Agent
from ..config import config, SimulationConfig

@pytest.fixture
def controller():
    """Create a random controller for testing."""
    return RandomController(max_velocity_change=1.0)

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig()

@pytest.fixture
def agent(test_config):
    """Create an agent for testing."""
    return Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)

def test_initialization(controller):
    """Test controller initialization."""
    assert controller.max_velocity_change == 1.0
    assert controller.action_space == (-1.0, 1.0)

def test_get_action(controller, agent):
    """Test action generation."""
    action = controller.get_action(agent)
    
    # Check action shape and constraints
    assert action.shape == (2,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

def test_apply_action(controller, agent):
    """Test action application."""
    # Test normal action application
    action = np.array([0.5, 0.5])
    controller.apply_action(agent, action)
    assert np.array_equal(agent.velocity, action)
    
    # Test velocity magnitude limit
    action = np.array([10.0, 10.0])
    controller.apply_action(agent, action)
    velocity_magnitude = np.linalg.norm(agent.velocity)
    assert velocity_magnitude <= agent.config.MAX_VELOCITY

def test_control_team(controller, test_config):
    """Test team control."""
    # Create a team of agents
    agents = [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 0, test_config)
    ]
    
    # Control the team
    controller.control_team(agents)
    
    # Check that all active agents have been controlled
    for agent in agents:
        if agent.is_active:
            velocity_magnitude = np.linalg.norm(agent.velocity)
            assert velocity_magnitude <= agent.config.MAX_VELOCITY

def test_inactive_agents(controller, test_config):
    """Test handling of inactive agents."""
    agent = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)
    agent.deactivate()
    
    # Control should not affect inactive agents
    initial_velocity = agent.velocity.copy()
    controller.control_team([agent])
    assert np.array_equal(agent.velocity, initial_velocity) 