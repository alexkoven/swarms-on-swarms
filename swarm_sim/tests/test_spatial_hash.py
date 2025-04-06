"""Tests for the spatial hash implementation."""

import pytest
import numpy as np
from ..environment.spatial_hash import SpatialHash
from ..environment.agent import Agent
from ..config import SimulationConfig

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return SimulationConfig(AGENT_RADIUS=5.0)  # Smaller agent radius for more precise testing

@pytest.fixture
def spatial_hash(test_config):
    """Create a test spatial hash."""
    return SpatialHash(test_config)

@pytest.fixture
def agents(test_config):
    """Create test agents."""
    return [
        Agent(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0, test_config),
        Agent(np.array([15.0, 15.0]), np.array([0.0, 0.0]), 1, test_config),  # Different bin from agent 0
        Agent(np.array([150.0, 150.0]), np.array([0.0, 0.0]), 0, test_config),  # Far from others
    ]

def test_initialization(spatial_hash, test_config):
    """Test spatial hash initialization."""
    assert spatial_hash.bin_size == test_config.AGENT_RADIUS * 2
    assert spatial_hash.num_bins_x == int(np.ceil(test_config.WORLD_SIZE[0] / spatial_hash.bin_size))
    assert spatial_hash.num_bins_y == int(np.ceil(test_config.WORLD_SIZE[1] / spatial_hash.bin_size))

def test_bin_indices(spatial_hash, test_config):
    """Test bin index calculation."""
    # Test center of world
    bin_index = spatial_hash._get_bin_index(np.array([test_config.WORLD_SIZE[0]/2, test_config.WORLD_SIZE[1]/2]))
    assert 0 <= bin_index[0] < spatial_hash.num_bins_x
    assert 0 <= bin_index[1] < spatial_hash.num_bins_y
    
    # Test world boundaries
    bin_index = spatial_hash._get_bin_index(np.array([0.0, 0.0]))
    assert bin_index == (0, 0)
    
    bin_index = spatial_hash._get_bin_index(np.array([test_config.WORLD_SIZE[0]-1, test_config.WORLD_SIZE[1]-1]))
    assert bin_index == (spatial_hash.num_bins_x - 1, spatial_hash.num_bins_y - 1)

def test_insert_and_remove(spatial_hash, agents):
    """Test agent insertion and removal."""
    # Insert agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Check agents are in correct bins
    for agent in agents:
        bin_index = spatial_hash._get_bin_index(agent.position)
        assert agent in spatial_hash.bins[bin_index].agents
    
    # Remove agents
    for agent in agents:
        spatial_hash.remove_agent(agent)
        bin_index = spatial_hash._get_bin_index(agent.position)
        assert agent not in spatial_hash.bins[bin_index].agents

def test_update(spatial_hash, agents):
    """Test agent position updates."""
    # Insert agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Update positions
    for agent in agents:
        old_pos = agent.position.copy()
        agent.position += np.array([10.0, 10.0])
        spatial_hash.update(agent)
        
        # Check agent is in new bin
        new_bin_index = spatial_hash._get_bin_index(agent.position)
        assert agent in spatial_hash.bins[new_bin_index].agents
        
        # Check agent is not in old bin
        old_bin_index = spatial_hash._get_bin_index(old_pos)
        assert agent not in spatial_hash.bins[old_bin_index].agents

def test_potential_collisions(spatial_hash, agents):
    """Test potential collision detection."""
    # Insert agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Get potential collisions for each agent
    for agent in agents:
        nearby = spatial_hash.get_potential_collisions(agent)
        # Should not include self
        assert agent not in nearby
        # Should include agents in same or adjacent bins
        for other in agents:
            if other is not agent:
                bin1 = spatial_hash._get_bin_index(agent.position)
                bin2 = spatial_hash._get_bin_index(other.position)
                if abs(bin1[0] - bin2[0]) <= 1 and abs(bin1[1] - bin2[1]) <= 1:
                    assert other in nearby
                else:
                    assert other not in nearby

def test_clear(spatial_hash, agents):
    """Test clearing the spatial hash."""
    # Insert agents
    for agent in agents:
        spatial_hash.insert(agent)
    
    # Clear spatial hash
    spatial_hash.clear()
    
    # Check all bins are empty
    for x in range(spatial_hash.num_bins_x):
        for y in range(spatial_hash.num_bins_y):
            bin_index = (x, y)
            assert len(spatial_hash.bins[bin_index].agents) == 0 