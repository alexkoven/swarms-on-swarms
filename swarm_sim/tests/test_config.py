"""Test configuration management functionality."""

import pytest
from ..config import SimulationConfig

def test_config_validation():
    """Test that configuration validation works correctly."""
    config = SimulationConfig()
    
    # Test valid configuration
    config.validate()
    
    # Test invalid world size
    config.WORLD_SIZE = -1
    with pytest.raises(AssertionError):
        config.validate()
    config.WORLD_SIZE = 1000.0  # Reset
    
    # Test invalid boundary type
    config.BOUNDARY_TYPE = "invalid"
    with pytest.raises(AssertionError):
        config.validate()
    config.BOUNDARY_TYPE = "wrap"  # Reset
    
    # Test invalid number of agents
    config.NUM_AGENTS = 0
    with pytest.raises(AssertionError):
        config.validate()
    config.NUM_AGENTS = 1000  # Reset 