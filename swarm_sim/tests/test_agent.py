"""Tests for the Agent class."""

import pytest
import numpy as np
from ..environment.agent import Agent
from ..config import config

def test_agent_initialization():
    """Test agent initialization with valid parameters."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])
    team_id = 0
    
    agent = Agent(position, velocity, team_id)
    
    assert np.array_equal(agent.position, position)
    assert np.array_equal(agent.velocity, velocity)
    assert agent.team_id == team_id
    assert agent.radius == config.AGENT_RADIUS
    assert agent.is_active

def test_agent_validation():
    """Test agent validation during initialization."""
    # Test invalid position dimension
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), 0)
    
    # Test invalid velocity dimension
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0)
    
    # Test invalid team_id
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), -1)
    
    # Test invalid radius
    with pytest.raises(AssertionError):
        Agent(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0, radius=-1.0)

def test_velocity_speed_limit():
    """Test velocity speed limit enforcement."""
    position = np.array([0.0, 0.0])
    velocity = np.array([10.0, 10.0])  # Speed > MAX_SPEED
    team_id = 0
    
    agent = Agent(position, velocity, team_id)
    expected_speed = config.MAX_SPEED
    actual_speed = np.linalg.norm(agent.velocity)
    
    assert np.isclose(actual_speed, expected_speed)

def test_position_update():
    """Test position update with different boundary conditions."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])
    team_id = 0
    dt = 1.0
    
    agent = Agent(position, velocity, team_id)
    
    # Test wrap-around boundary
    config.BOUNDARY_TYPE = "wrap"
    agent.update_position(dt)
    assert np.array_equal(agent.position, np.array([1.0, 1.0]))
    
    # Test bounce boundary
    config.BOUNDARY_TYPE = "bounce"
    agent.position = np.array([0.0, 0.0])
    agent.velocity = np.array([-1.0, -1.0])
    agent.update_position(dt)
    assert np.array_equal(agent.position, np.array([0.0, 0.0]))
    assert np.array_equal(agent.velocity, np.array([1.0, 1.0]))

def test_set_velocity():
    """Test velocity setting with speed limit enforcement."""
    agent = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    
    # Test setting velocity within speed limit
    new_velocity = np.array([1.0, 1.0])
    agent.set_velocity(new_velocity)
    assert np.array_equal(agent.velocity, new_velocity)
    
    # Test setting velocity exceeding speed limit
    fast_velocity = np.array([10.0, 10.0])
    agent.set_velocity(fast_velocity)
    assert np.isclose(np.linalg.norm(agent.velocity), config.MAX_SPEED)

def test_get_state():
    """Test getting agent state."""
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    team_id = 0
    
    agent = Agent(position, velocity, team_id)
    state_position, state_velocity, state_team_id = agent.get_state()
    
    assert np.array_equal(state_position, position)
    assert np.array_equal(state_velocity, velocity)
    assert state_team_id == team_id
    
    # Verify that returned arrays are copies
    state_position[0] = 999.0
    assert not np.array_equal(state_position, agent.position)

def test_deactivate():
    """Test agent deactivation."""
    agent = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    assert agent.is_active
    
    agent.deactivate()
    assert not agent.is_active 