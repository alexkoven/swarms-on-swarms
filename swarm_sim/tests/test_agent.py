"""Tests for the Agent class."""

import pytest
import numpy as np
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

def test_initialization(agent, test_config):
    """Test agent initialization."""
    assert np.array_equal(agent.position, np.array([0.0, 0.0]))
    assert np.array_equal(agent.velocity, np.array([0.0, 0.0]))
    assert agent.team_id == 0
    assert agent.config == test_config
    assert agent.radius == test_config.AGENT_RADIUS
    assert agent.is_active

def test_agent_validation(test_config):
    """Test agent validation."""
    # Test invalid position dimension
    with pytest.raises(AssertionError, match="Position must be a 2D vector"):
        Agent(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), 0, test_config)
    
    # Test invalid velocity dimension
    with pytest.raises(AssertionError, match="Velocity must be a 2D vector"):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0, test_config)
    
    # Test invalid team ID
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), test_config.NUM_TEAMS, test_config)
    
    # Test invalid radius
    with pytest.raises(AssertionError, match="Agent radius must be positive"):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0, test_config, radius=0.0)

def test_velocity_speed_limit(agent, test_config):
    """Test velocity speed limit enforcement."""
    # Test velocity exceeding max speed
    fast_velocity = np.array([100.0, 100.0])
    agent.set_velocity(fast_velocity)
    speed = np.linalg.norm(agent.velocity)
    assert speed <= test_config.MAX_VELOCITY
    
    # Test velocity within limit
    slow_velocity = np.array([1.0, 1.0])
    agent.set_velocity(slow_velocity)
    assert np.array_equal(agent.velocity, slow_velocity)

def test_update_position(agent, test_config):
    """Test position update."""
    # Set initial position and velocity
    agent.position = np.array([0.0, 0.0])
    agent.velocity = np.array([1.0, 1.0])
    
    # Update position
    agent.update_position(0.1)
    assert np.array_equal(agent.position, np.array([0.1, 0.1]))
    
    # Test boundary wrapping
    agent.position = np.array([test_config.WORLD_SIZE[0] - 1.0, test_config.WORLD_SIZE[1] - 1.0])
    agent.velocity = np.array([2.0, 2.0])
    agent.update_position(0.1)
    assert agent.position[0] < test_config.WORLD_SIZE[0]
    assert agent.position[1] < test_config.WORLD_SIZE[1]

def test_set_velocity(agent, test_config):
    """Test velocity setting."""
    # Test normal velocity
    velocity = np.array([1.0, 1.0])
    agent.set_velocity(velocity)
    assert np.array_equal(agent.velocity, velocity)
    
    # Test zero velocity
    agent.set_velocity(np.array([0.0, 0.0]))
    assert np.array_equal(agent.velocity, np.array([0.0, 0.0]))
    
    # Test velocity exceeding max speed
    fast_velocity = np.array([100.0, 100.0])
    agent.set_velocity(fast_velocity)
    speed = np.linalg.norm(agent.velocity)
    assert speed <= test_config.MAX_VELOCITY

def test_get_state(agent, test_config):
    """Test getting agent state."""
    # Set initial state
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    team_id = 0
    
    agent.position = position
    agent.velocity = velocity
    agent.team_id = team_id
    
    # Get state
    state = agent.get_state()
    assert isinstance(state, tuple)
    assert len(state) == 3
    assert np.array_equal(state[0], position)
    assert np.array_equal(state[1], velocity)
    assert state[2] == team_id

def test_deactivate(agent):
    """Test agent deactivation."""
    assert agent.is_active
    agent.deactivate()
    assert not agent.is_active 