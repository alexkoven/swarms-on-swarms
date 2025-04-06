"""Example usage of the swarm simulation.

This module provides examples of how to use the swarm simulation package,
demonstrating various features and use cases.
"""

import numpy as np
from .environment.swarm_env import SwarmEnv
from .controllers.random_controller import RandomController
from .config import SimulationConfig

def basic_simulation():
    """Run a basic simulation with default settings.
    
    This example shows how to:
    1. Create a simulation environment
    2. Run the simulation for a fixed number of steps
    3. Access simulation state
    """
    # Create configuration
    config = SimulationConfig.default()
    
    # Create environment
    env = SwarmEnv(config)
    
    # Create controller
    controller = RandomController()
    
    # Run simulation
    num_steps = 1000
    for step in range(num_steps):
        # Get agents for each team
        for team_id in range(config.NUM_TEAMS):
            team_agents = env.get_team_agents(team_id)
            # Control agents
            controller.control_team(team_agents)
        
        # Step environment
        env.step()
        
        # Print status every 100 steps
        if step % 100 == 0:
            state = env.get_state()
            print(f"Step {step}: Active agents per team: {state.team_counts}")

def custom_configuration():
    """Demonstrate custom configuration options.
    
    This example shows how to:
    1. Create a custom configuration
    2. Modify simulation parameters
    3. Run simulation with custom settings
    """
    # Create custom configuration
    config = SimulationConfig(
        WORLD_SIZE=(2000.0, 2000.0),  # Larger world
        TIME_STEP=0.033,  # 30 FPS
        AGENT_RADIUS=10.0,  # Larger agents
        MAX_VELOCITY=150.0,  # Faster agents
        MAX_VELOCITY_CHANGE=15.0,  # More aggressive movement
        NUM_TEAMS=3,  # Three teams
        AGENTS_PER_TEAM=500,  # Fewer agents per team
        TEAM_COLORS=('red', 'blue', 'green'),  # Three team colors
        FRAME_RATE=30,  # Lower frame rate
        VISUALIZE=True  # Enable visualization
    )
    
    # Create and run simulation
    env = SwarmEnv(config)
    controller = RandomController()
    
    # Run for 1000 steps
    for _ in range(1000):
        for team_id in range(config.NUM_TEAMS):
            controller.control_team(env.get_team_agents(team_id))
        env.step()

def performance_benchmark():
    """Run a performance benchmark with different swarm sizes.
    
    This example shows how to:
    1. Test performance with different swarm sizes
    2. Measure step computation time
    3. Monitor memory usage
    """
    import time
    import psutil
    
    def measure_memory():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # Test different swarm sizes
    swarm_sizes = [100, 500, 1000, 2000]
    results = []
    
    for size in swarm_sizes:
        # Create configuration
        config = SimulationConfig(
            AGENTS_PER_TEAM=size,
            VISUALIZE=False  # Disable visualization for benchmarking
        )
        
        # Create environment
        env = SwarmEnv(config)
        controller = RandomController()
        
        # Warm up
        for _ in range(100):
            for team_id in range(config.NUM_TEAMS):
                controller.control_team(env.get_team_agents(team_id))
            env.step()
        
        # Measure performance
        start_time = time.time()
        start_memory = measure_memory()
        
        # Run benchmark
        num_steps = 1000
        for _ in range(num_steps):
            for team_id in range(config.NUM_TEAMS):
                controller.control_team(env.get_team_agents(team_id))
            env.step()
        
        end_time = time.time()
        end_memory = measure_memory()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_step_time = (total_time / num_steps) * 1000  # ms
        memory_used = end_memory - start_memory
        
        results.append({
            'swarm_size': size,
            'avg_step_time_ms': avg_step_time,
            'memory_used_mb': memory_used
        })
        
        print(f"\nResults for {size} agents per team:")
        print(f"Average step time: {avg_step_time:.2f} ms")
        print(f"Memory used: {memory_used:.2f} MB")
    
    return results

if __name__ == '__main__':
    print("Running basic simulation example...")
    basic_simulation()
    
    print("\nRunning custom configuration example...")
    custom_configuration()
    
    print("\nRunning performance benchmark...")
    performance_benchmark() 