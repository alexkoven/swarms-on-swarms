"""Test configuration management functionality."""

import pytest
from ..config import SimulationConfig

def test_config_validation():
    """Test that configuration validation works correctly."""
    # Test valid configuration
    config = SimulationConfig()
    config.validate()  # Should not raise any exceptions

    # Test invalid world size
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(WORLD_SIZE=(0, 100))
        invalid_config.validate()

    # Test invalid time step
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(TIME_STEP=0)
        invalid_config.validate()

    # Test invalid agent radius
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(AGENT_RADIUS=-1)
        invalid_config.validate()

    # Test invalid number of teams
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(NUM_TEAMS=0)
        invalid_config.validate()

    # Test invalid boundary type
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(BOUNDARY_TYPE="invalid")
        invalid_config.validate()

    # Test mismatched team colors
    with pytest.raises(AssertionError):
        invalid_config = SimulationConfig(NUM_TEAMS=3, TEAM_COLORS=("red", "blue"))
        invalid_config.validate() 