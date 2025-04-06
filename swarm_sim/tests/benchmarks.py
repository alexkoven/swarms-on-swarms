"""Performance benchmarks for the swarm simulation.

This module contains functions to benchmark the simulation with different swarm sizes
and configurations. It measures:
- Step computation time
- Memory usage
- Collision detection performance
- Overall simulation performance
"""

import time
import psutil
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from statistics import mean, stdev

from ..environment.swarm_env import SwarmEnv
from ..config import SimulationConfig

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    swarm_size: int
    step_times: List[float]
    memory_usage: List[float]
    collision_checks: List[int]
    active_agents: List[int]

def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(
    swarm_size: int,
    num_steps: int = 100,
    num_runs: int = 5
) -> BenchmarkResult:
    """Run benchmark with specified swarm size.
    
    Args:
        swarm_size: Number of agents per team
        num_steps: Number of simulation steps per run
        num_runs: Number of benchmark runs to average
        
    Returns:
        BenchmarkResult containing performance metrics
    """
    step_times = []
    memory_usage = []
    collision_checks = []
    active_agents = []
    
    for _ in range(num_runs):
        # Create environment with specified swarm size
        config = SimulationConfig()
        config.AGENTS_PER_TEAM = swarm_size
        env = SwarmEnv(config)
        
        # Warm up simulation
        for _ in range(10):
            env.step()
        
        # Run benchmark
        run_step_times = []
        run_memory = []
        run_collisions = []
        run_active = []
        
        for _ in range(num_steps):
            start_time = time.time()
            start_memory = measure_memory_usage()
            
            # Run step and collect metrics
            env.step()
            
            end_time = time.time()
            end_memory = measure_memory_usage()
            
            run_step_times.append(end_time - start_time)
            run_memory.append(end_memory - start_memory)
            run_collisions.append(env.spatial_hash.collision_checks)
            run_active.append(len([a for a in env.agents if a.is_active]))
        
        step_times.extend(run_step_times)
        memory_usage.extend(run_memory)
        collision_checks.extend(run_collisions)
        active_agents.extend(run_active)
    
    return BenchmarkResult(
        swarm_size=swarm_size,
        step_times=step_times,
        memory_usage=memory_usage,
        collision_checks=collision_checks,
        active_agents=active_agents
    )

def run_swarm_size_benchmarks(
    swarm_sizes: List[int] = [100, 500],  # Reduced to just 100 and 500
    num_steps: int = 100,
    num_runs: int = 5
) -> Dict[int, BenchmarkResult]:
    """Run benchmarks with different swarm sizes.
    
    Args:
        swarm_sizes: List of swarm sizes to benchmark
        num_steps: Number of simulation steps per run
        num_runs: Number of benchmark runs to average
        
    Returns:
        Dictionary mapping swarm sizes to benchmark results
    """
    results = {}
    for size in swarm_sizes:
        print(f"Running benchmark with {size} agents per team...")
        results[size] = run_benchmark(size, num_steps, num_runs)
    return results

def print_benchmark_results(results: Dict[int, BenchmarkResult]):
    """Print formatted benchmark results.
    
    Args:
        results: Dictionary mapping swarm sizes to benchmark results
    """
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Swarm Size':<15} {'Avg Step Time (ms)':<20} {'Memory (MB)':<15} {'Collision Checks':<20} {'Active Agents'}")
    print("-" * 80)
    
    for size, result in results.items():
        avg_step_time = mean(result.step_times) * 1000
        avg_memory = mean(result.memory_usage)
        avg_collisions = mean(result.collision_checks)
        avg_active = mean(result.active_agents)
        
        print(f"{size:<15} {avg_step_time:>10.2f} ms    {avg_memory:>10.2f} MB    {avg_collisions:>10.0f}    {avg_active:>10.0f}")

if __name__ == "__main__":
    # Run benchmarks with different swarm sizes
    results = run_swarm_size_benchmarks()
    print_benchmark_results(results) 