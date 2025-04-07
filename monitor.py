import psutil
import time
import os
import sys
from swarm_sim.environment.swarm_env import SwarmEnv
from swarm_sim.config import SimulationConfig

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def main():
    # Create configuration
    config = SimulationConfig.default()
    env = SwarmEnv(config)
    
    print("Starting simulation with monitoring...")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Run simulation with monitoring
    for step in range(1000):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Run step
        state = env.step()
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Print diagnostics every 100 steps
        if step % 100 == 0:
            print(f"\nStep {step}:")
            print(f"Step time: {(end_time - start_time) * 1000:.1f} ms")
            print(f"Memory change: {end_memory - start_memory:.2f} MB")
            print(f"Total memory: {end_memory:.2f} MB")
            print(f"Active agents: {sum(state.team_counts.values())}")
            print(f"Collision checks: {env.spatial_hash.collision_checks}")
            print("-" * 50)

if __name__ == "__main__":
    main() 