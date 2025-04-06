"""Tests for the SpatialHash class."""

import pytest
import numpy as np
from ..environment.spatial_hash import SpatialHash
from ..environment.agent import Agent
from ..config import config

@pytest.fixture
def spatial_hash():
    """Create a spatial hash instance for testing."""
    return SpatialHash(bin_size=10.0)  # Smaller bin size for more precise testing

@pytest.fixture
def agents():
    """Create test agents."""
    return [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0),
        Agent(np.array([15.0, 15.0]), np.array([0.0, 0.0]), 1),  # Different bin from agent 0
        Agent(np.array([150.0, 150.0]), np.array([0.0, 0.0]), 0),  # Far from others
    ]

def test_initialization(spatial_hash):
    """Test spatial hash initialization."""
    assert spatial_hash.bin_size == 10.0
    assert spatial_hash.num_bins == int(np.ceil(config.WORLD_SIZE / 10.0))
    assert len(spatial_hash.bins) == spatial_hash.num_bins
    assert len(spatial_hash.bins[0]) == spatial_hash.num_bins

def test_bin_indices(spatial_hash):
    """Test bin index calculation."""
    # Test center of first bin
    i, j = spatial_hash._get_bin_indices(np.array([0.0, 0.0]))
    assert i == 0 and j == 0
    
    # Test center of second bin
    i, j = spatial_hash._get_bin_indices(np.array([15.0, 15.0]))
    assert i == 1 and j == 1
    
    # Test center of last bin
    i, j = spatial_hash._get_bin_indices(np.array([990.0, 990.0]))
    assert i == 99 and j == 99
    
    # Test wrap-around
    i, j = spatial_hash._get_bin_indices(np.array([1000.0, 1000.0]))
    assert i == 0 and j == 0

def test_insert_and_remove(spatial_hash, agents):
    """Test agent insertion and removal."""
    # Test insertion
    for agent in agents:
        spatial_hash.insert(agent)
        i, j = spatial_hash._get_bin_indices(agent.position)
        assert agent in spatial_hash.bins[i][j]
    
    # Test removal
    for agent in agents:
        spatial_hash.remove(agent)
        i, j = spatial_hash._get_bin_indices(agent.position)
        assert agent not in spatial_hash.bins[i][j]

def test_update(spatial_hash, agents):
    """Test agent position update."""
    agent = agents[0]
    old_position = agent.position.copy()
    spatial_hash.insert(agent)
    
    # Update position
    agent.position = np.array([25.0, 25.0])  # Move to a different bin
    spatial_hash.update(agent)
    
    # Check old position's bin
    old_i, old_j = spatial_hash._get_bin_indices(old_position)
    assert agent not in spatial_hash.bins[old_i][old_j]
    
    # Check new position's bin
    new_i, new_j = spatial_hash._get_bin_indices(agent.position)
    assert agent in spatial_hash.bins[new_i][new_j]

def test_potential_collisions(spatial_hash, agents):
    """Test potential collision detection."""
    # Insert all agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Get potential collisions for first agent
    potential = spatial_hash.get_potential_collisions(agents[0])
    
    # Only agent[1] should be in potential collisions if it's in an adjacent bin
    # agent[2] is too far away to be in potential collisions
    assert len(potential) == 0  # No agents in adjacent bins
    
    # Move agent[1] closer to agent[0]
    agents[1].position = np.array([5.0, 5.0])  # Same bin as agent[0]
    spatial_hash.update(agents[1])
    
    potential = spatial_hash.get_potential_collisions(agents[0])
    assert len(potential) == 1
    assert agents[1] in potential
    assert agents[2] not in potential

def test_collision_detection(spatial_hash, agents):
    """Test collision detection between agents."""
    # Test no collision (different teams, far apart)
    assert not spatial_hash.check_collision(agents[0], agents[1])
    
    # Test no collision (same team)
    assert not spatial_hash.check_collision(agents[0], agents[2])
    
    # Test collision (different teams, close together)
    agent1 = Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0)
    agent2 = Agent(np.array([2.0, 2.0]), np.array([0.0, 0.0]), 1)
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
    for i in range(spatial_hash.num_bins):
        for j in range(spatial_hash.num_bins):
            assert len(spatial_hash.bins[i][j]) == 0 