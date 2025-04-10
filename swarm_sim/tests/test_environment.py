"""Tests for the SwarmEnv class."""

import pytest
import numpy as np
from ..environment.swarm_env import SwarmEnv
from ..environment.agent import Agent
from ..config import SimulationConfig

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig(
        TIME_STEP=0.1,
        AGENTS_PER_TEAM=0  # No agents for testing
    )

@pytest.fixture
def env(test_config):
    """Create a simulation environment for testing."""
    return SwarmEnv(test_config)

def test_initialization(env, test_config):
    """Test environment initialization."""
    assert env.config.TIME_STEP == test_config.TIME_STEP
    assert env.step_count == 0
    assert len(env.agents) == 0
    assert all(count == 0 for count in env.team_counts.values())

def test_add_agent(env, test_config):
    """Test agent addition."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])
    team_id = 0
    
    agent = env.add_agent(position, velocity, team_id)
    
    assert agent in env.agents
    assert env.team_counts[team_id] == 1
    assert np.array_equal(agent.position, position)
    assert np.array_equal(agent.velocity, velocity)
    assert agent.team_id == team_id

def test_remove_agent(env, test_config):
    """Test agent removal."""
    agent = env.add_agent(np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0)
    assert env.team_counts[0] == 1
    
    env.remove_agent(agent)
    assert agent not in env.agents
    assert env.team_counts[0] == 0
    assert not agent.is_active

def test_get_team_agents(env, test_config):
    """Test getting agents by team."""
    # Add agents to different teams
    agent1 = env.add_agent(np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0)
    agent2 = env.add_agent(np.array([10.0, 10.0]), np.array([1.0, 1.0]), 0)
    agent3 = env.add_agent(np.array([20.0, 20.0]), np.array([1.0, 1.0]), 1)
    
    team0_agents = env.get_team_agents(0)
    team1_agents = env.get_team_agents(1)
    
    assert len(team0_agents) == 2
    assert len(team1_agents) == 1
    assert agent1 in team0_agents
    assert agent2 in team0_agents
    assert agent3 in team1_agents

def test_get_nearby_agents(env, test_config):
    """Test getting nearby agents."""
    agent1 = env.add_agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    agent2 = env.add_agent(np.array([5.0, 5.0]), np.array([0.0, 0.0]), 1)  # Close to agent1
    agent3 = env.add_agent(np.array([100.0, 100.0]), np.array([0.0, 0.0]), 1)  # Far from agent1
    
    nearby = env.get_nearby_agents(agent1)
    assert agent2 in nearby  # Should be nearby
    assert agent3 not in nearby  # Should be too far

def test_collision_detection(env, test_config):
    """Test collision detection between agents."""
    agent1 = env.add_agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    agent2 = env.add_agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 1)  # Close enough to collide
    agent3 = env.add_agent(np.array([100.0, 100.0]), np.array([0.0, 0.0]), 1)  # Too far to collide
    
    collisions = env.spatial_hash.get_potential_collisions(agent1)
    assert agent2 in collisions
    assert agent3 not in collisions

def test_handle_collisions(env, test_config):
    """Test collision handling."""
    agent1 = env.add_agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    agent2 = env.add_agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 1)
    
    collisions = env.handle_collisions()
    assert len(collisions) == 1
    assert (agent1, agent2) in collisions or (agent2, agent1) in collisions
    assert not agent1.is_active
    assert not agent2.is_active
    assert env.team_counts[0] == 0
    assert env.team_counts[1] == 0

def test_step(env, test_config):
    """Test simulation step."""
    # Add two agents on collision course
    agent1 = env.add_agent(np.array([0.0, 0.0]), np.array([10.0, 10.0]), 0)
    agent2 = env.add_agent(np.array([10.0, 10.0]), np.array([-10.0, -10.0]), 1)
    
    # Run a few steps
    for _ in range(5):
        state = env.step()
        if not agent1.is_active or not agent2.is_active:
            assert state.team_counts[0] == 0 or state.team_counts[1] == 0
            break
    
    assert env.step_count > 0

def test_reset(env, test_config):
    """Test environment reset."""
    # Add some agents and run steps
    env.add_agent(np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0)
    env.add_agent(np.array([10.0, 10.0]), np.array([1.0, 1.0]), 1)
    env.step()
    
    # Reset environment
    env.reset()
    assert env.step_count == 0
    assert len(env.agents) == 0
    assert all(count == 0 for count in env.team_counts.values())

def test_get_state(env, test_config):
    """Test getting environment state."""
    agent = env.add_agent(np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0)
    env.step()
    
    state = env.get_state()
    assert state.step_count == 1
    assert state.team_counts[0] == 1
    assert len(state.active_agents) == 1
    
    agent_state = state.active_agents[0]
    assert np.array_equal(agent_state.position, agent.position)
    assert np.array_equal(agent_state.velocity, agent.velocity)
    assert agent_state.team_id == agent.team_id
    assert agent_state.is_active == agent.is_active 