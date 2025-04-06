"""Tests for the Agent class."""

import pytest
import numpy as np
from ..environment.agent import Agent
from ..config import config, SimulationConfig

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig()

@pytest.fixture
def agent(test_config):
    """Create a test agent."""
    return Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)

def test_initialization(test_config):
    """Test agent initialization."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])
    team_id = 0
    
    agent = Agent(position, velocity, team_id, test_config)
    
    assert np.array_equal(agent.position, position)
    assert np.array_equal(agent.velocity, velocity)
    assert agent.team_id == team_id
    assert agent.radius == test_config.AGENT_RADIUS
    assert agent.is_active

def test_agent_validation(test_config):
    """Test agent validation during initialization."""
    # Test invalid position dimension
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), 0, test_config)
    
    # Test invalid velocity dimension
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0, test_config)
    
    # Test invalid team_id
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), -1, test_config)
    
    # Test invalid radius
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0, test_config, radius=-1.0)

def test_velocity_speed_limit(test_config):
    """Test velocity speed limit enforcement."""
    position = np.array([0.0, 0.0])
    velocity = np.array([100.0, 100.0])  # Speed > MAX_VELOCITY
    team_id = 0

    agent = Agent(position, velocity, team_id, test_config)
    expected_speed = test_config.MAX_VELOCITY
    actual_speed = np.linalg.norm(agent.velocity)

    assert np.isclose(actual_speed, expected_speed, rtol=1e-10)

def test_update_position(test_config):
    """Test position update with different boundary conditions."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])
    team_id = 0
    dt = 1.0
    
    agent = Agent(position, velocity, team_id, test_config)
    
    # Test wrap-around boundary
    test_config.BOUNDARY_TYPE = "wrap"
    agent.update_position(dt)
    assert np.array_equal(agent.position, np.array([1.0, 1.0]))
    
    # Test bounce boundary
    test_config.BOUNDARY_TYPE = "bounce"
    agent.position = np.array([0.0, 0.0])
    agent.velocity = np.array([-1.0, -1.0])
    agent.update_position(dt)
    assert np.array_equal(agent.position, np.array([0.0, 0.0]))
    assert np.array_equal(agent.velocity, np.array([1.0, 1.0]))  # Velocity should be reversed

def test_set_velocity(test_config):
    """Test velocity setting with speed limit enforcement."""
    agent = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)

    # Test setting velocity within speed limit
    new_velocity = np.array([1.0, 1.0])
    agent.set_velocity(new_velocity)
    assert np.array_equal(agent.velocity, new_velocity)

    # Test setting velocity exceeding speed limit
    fast_velocity = np.array([100.0, 100.0])
    agent.set_velocity(fast_velocity)
    assert np.isclose(np.linalg.norm(agent.velocity), test_config.MAX_VELOCITY, rtol=1e-10)

def test_get_state(test_config):
    """Test getting agent state."""
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    team_id = 0
    
    agent = Agent(position, velocity, team_id, test_config)
    state_position, state_velocity, state_team_id = agent.get_state()
    
    assert np.array_equal(state_position, position)
    assert np.array_equal(state_velocity, velocity)
    assert state_team_id == team_id
    
    # Verify that returned arrays are copies
    state_position[0] = 999.0
    assert not np.array_equal(state_position, agent.position)

def test_deactivate(test_config):
    """Test agent deactivation."""
    agent = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)
    assert agent.is_active
    
    agent.deactivate()
    assert not agent.is_active 