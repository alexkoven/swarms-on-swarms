"""Tests for the SpatialHash class."""

import pytest
import numpy as np
from ..environment.spatial_hash import SpatialHash
from ..environment.agent import Agent
from ..config import config, SimulationConfig

@pytest.fixture
def spatial_hash():
    """Create a spatial hash instance for testing."""
    test_config = SimulationConfig(AGENT_RADIUS=5.0)  # Smaller agent radius for more precise testing
    return SpatialHash(test_config)

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig(AGENT_RADIUS=5.0)

@pytest.fixture
def agents(test_config):
    """Create test agents."""
    return [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([15.0, 15.0]), np.array([0.0, 0.0]), 1, test_config),  # Different bin from agent 0
        Agent(np.array([150.0, 150.0]), np.array([0.0, 0.0]), 0, test_config),  # Far from others
    ]

def test_initialization(spatial_hash):
    """Test spatial hash initialization."""
    assert spatial_hash.bin_size == spatial_hash.config.AGENT_RADIUS * 2
    assert spatial_hash.num_bins_x == int(np.ceil(spatial_hash.config.WORLD_SIZE[0] / spatial_hash.bin_size))
    assert spatial_hash.num_bins_y == int(np.ceil(spatial_hash.config.WORLD_SIZE[1] / spatial_hash.bin_size))
    assert len(spatial_hash.bins) == spatial_hash.num_bins_x * spatial_hash.num_bins_y

def test_bin_indices(spatial_hash):
    """Test bin index calculation."""
    # Test center of first bin
    bin_index = spatial_hash._get_bin_index(np.array([0.0, 0.0]))
    assert bin_index == (0, 0)
    
    # Test center of second bin
    bin_index = spatial_hash._get_bin_index(np.array([15.0, 15.0]))
    assert bin_index == (1, 1)
    
    # Test center of last bin
    bin_index = spatial_hash._get_bin_index(np.array([990.0, 990.0]))
    assert bin_index == (spatial_hash.num_bins_x - 1, spatial_hash.num_bins_y - 1)

def test_insert_and_remove(spatial_hash, agents):
    """Test agent insertion and removal."""
    # Test insertion
    for agent in agents:
        spatial_hash.insert(agent)
        bin_index = spatial_hash._get_bin_index(agent.position)
        assert agent in spatial_hash.bins[bin_index].agents
    
    # Test removal
    for agent in agents:
        spatial_hash.remove(agent)
        bin_index = spatial_hash._get_bin_index(agent.position)
        assert agent not in spatial_hash.bins[bin_index].agents

def test_update(spatial_hash, agents):
    """Test agent position update."""
    agent = agents[0]
    old_position = agent.position.copy()
    spatial_hash.insert(agent)
    
    # Update position
    agent.position = np.array([25.0, 25.0])  # Move to a different bin
    spatial_hash.update(agent)
    
    # Check old position's bin
    old_bin_index = spatial_hash._get_bin_index(old_position)
    assert agent not in spatial_hash.bins[old_bin_index].agents
    
    # Check new position's bin
    new_bin_index = spatial_hash._get_bin_index(agent.position)
    assert agent in spatial_hash.bins[new_bin_index].agents

def test_potential_collisions(spatial_hash, agents):
    """Test potential collision detection."""
    # Insert all agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Get potential collisions for first agent
    potential = spatial_hash.get_potential_collisions(agents[0])
    
    # Only agent[1] should be in potential collisions if it's in an adjacent bin
    # agent[2] is too far away to be in potential collisions
    assert len(potential) == 1  # Agent 1 is in an adjacent bin
    assert agents[1] in potential
    assert agents[2] not in potential

def test_collision_detection(spatial_hash, agents, test_config):
    """Test collision detection between agents."""
    # Test no collision (different teams, far apart)
    assert not spatial_hash.check_collision(agents[0], agents[1])
    
    # Test no collision (same team)
    assert not spatial_hash.check_collision(agents[0], agents[2])
    
    # Test collision (different teams, close together)
    agent1 = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config)
    agent2 = Agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 1, test_config)
    assert spatial_hash.check_collision(agent1, agent2)
    
    # Test inactive agent
    agent1.is_active = False
    assert not spatial_hash.check_collision(agent1, agent2)

def test_clear(spatial_hash, agents):
    """Test clearing all agents from spatial hash."""
    # Insert all agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Clear spatial hash
    spatial_hash.clear()
    
    # Verify all bins are empty
    for bin_index in spatial_hash.bins:
        assert len(spatial_hash.bins[bin_index].agents) == 0 