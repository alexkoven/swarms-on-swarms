"""Main entry point for the swarm simulation.

This module provides the main entry point for running the swarm simulation,
including visualization and command-line argument parsing.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from typing import Dict, List, Optional, Tuple

from .environment.swarm_env import SwarmEnv
from .controllers.random_controller import RandomController
from .config import SimulationConfig

class SwarmSimulation:
    """Main simulation class handling visualization and control.
    
    This class manages the simulation environment, controllers, and visualization.
    It provides methods for running the simulation with or without visualization,
    and handles the animation of agent positions and status updates.
    
    Attributes:
        env (SwarmEnv): The swarm environment
        controllers (Dict[int, RandomController]): Controllers for each team
        fig (plt.Figure): Matplotlib figure for visualization (if enabled)
        ax (plt.Axes): Matplotlib axes for plotting (if enabled)
        scatter_plots (Dict[int, plt.scatter]): Scatter plots for each team (if enabled)
        status_text (plt.Text): Text object for displaying simulation status (if enabled)
    
    The visualization includes:
    - Scatter plots for each team's agents
    - A status panel showing step count, active agents, and performance metrics
    - Dark theme for better visibility
    - Real-time updates at the configured frame rate
    """
    
    def __init__(self, env: SwarmEnv):
        """Initialize the simulation.
        
        Args:
            env (SwarmEnv): The swarm environment to simulate
        """
        self.env = env
        self.controllers = {
            team_id: RandomController() 
            for team_id in range(env.config.NUM_TEAMS)
        }
        
        # Setup visualization
        if env.config.VISUALIZE:
            plt.style.use('dark_background')
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(0, env.config.WORLD_SIZE[0])
            self.ax.set_ylim(0, env.config.WORLD_SIZE[1])
            self.ax.set_aspect('equal')
            self.ax.set_title('Swarm vs Swarm Simulation')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            
            # Create scatter plots for each team
            self.scatter_plots = {}
            for team_id in range(env.config.NUM_TEAMS):
                self.scatter_plots[team_id] = self.ax.scatter(
                    [], [], 
                    c=env.config.TEAM_COLORS[team_id % len(env.config.TEAM_COLORS)],
                    label=f'Team {team_id}',
                    alpha=0.8,
                    s=50
                )
            self.ax.legend(loc='upper right')
            
            # Add status text
            self.status_text = self.ax.text(
                0.02, 0.98, '', 
                transform=self.ax.transAxes,
                verticalalignment='top',
                color='white'
            )
    
    def update_visualization(self, frame: int) -> List[plt.Artist]:
        """Update the visualization for animation.
        
        Args:
            frame (int): Current frame number
            
        Returns:
            List[plt.Artist]: List of artists that were modified
        """
        # Update simulation
        start_time = time.time()
        for team_id, controller in self.controllers.items():
            team_agents = self.env.get_team_agents(team_id)
            controller.control_team(team_agents)
        
        state = self.env.step()
        step_time = time.time() - start_time
        
        # Update visualization
        artists = []
        for team_id, scatter in self.scatter_plots.items():
            team_agents = self.env.get_team_agents(team_id)
            active_agents = [agent for agent in team_agents if agent.is_active]
            
            if active_agents:
                positions = np.array([agent.position for agent in active_agents])
                scatter.set_offsets(positions)
            else:
                scatter.set_offsets(np.zeros((0, 2)))
            artists.append(scatter)
        
        # Update status text
        status = f'Step: {self.env.step_count}\n'
        for team_id in range(self.env.config.NUM_TEAMS):
            status += f'Team {team_id}: {state.team_counts[team_id]} active\n'
        status += f'Step time: {step_time*1000:.1f}ms'
        self.status_text.set_text(status)
        artists.append(self.status_text)
        
        return artists
    
    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the simulation.
        
        Args:
            num_steps (Optional[int]): Number of steps to run, or None for infinite
        """
        if self.env.config.VISUALIZE:
            interval = 1000 / self.env.config.FRAME_RATE  # Convert FPS to milliseconds
            anim = FuncAnimation(
                self.fig, 
                self.update_visualization,
                frames=num_steps,
                interval=interval,
                blit=True
            )
            plt.show()
        else:
            steps = num_steps if num_steps is not None else self.env.config.MAX_STEPS
            for _ in range(steps):
                for team_id, controller in self.controllers.items():
                    team_agents = self.env.get_team_agents(team_id)
                    controller.control_team(team_agents)
                self.env.step()

def create_visualization(env: SwarmEnv) -> Tuple[plt.Figure, plt.Axes]:
    """Create the visualization figure and axes.
    
    Args:
        env: Simulation environment
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.config.WORLD_SIZE[0])
    ax.set_ylim(0, env.config.WORLD_SIZE[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    # Add title and status panel
    title = ax.set_title('Swarm Simulation', color='white')
    status = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                    verticalalignment='top')
    
    return fig, ax, title, status

def update_visualization(frame: int, env: SwarmEnv, ax: plt.Axes,
                        title: plt.Text, status: plt.Text,
                        scatter_plots: List[plt.scatter]) -> List[plt.scatter]:
    """Update the visualization for the current frame.
    
    Args:
        frame: Current frame number
        env: Simulation environment
        ax: Matplotlib axes
        title: Title text object
        status: Status text object
        scatter_plots: List of scatter plots for each team
        
    Returns:
        Updated scatter plots
    """
    # Run simulation step
    start_time = time.time()
    state = env.step()
    step_time = time.time() - start_time
    
    # Update title with frame count
    title.set_text(f'Swarm Simulation - Step {frame}')
    
    # Update status panel
    status_text = (
        f'Step: {frame}\n'
        f'Active Agents:\n'
        f'  Team 0: {state.team_counts[0]}\n'
        f'  Team 1: {state.team_counts[1]}\n'
        f'Step Time: {step_time*1000:.2f} ms'
    )
    status.set_text(status_text)
    
    # Update agent positions
    for team_id, scatter in enumerate(scatter_plots):
        team_agents = [a for a in env.agents if a.team_id == team_id and a.is_active]
        if team_agents:
            positions = np.array([a.position for a in team_agents])
            scatter.set_offsets(positions)
        else:
            scatter.set_offsets(np.array([]))
    
    return scatter_plots

def run_simulation(env: SwarmEnv, num_steps: int = None) -> None:
    """Run the simulation with visualization.
    
    Args:
        env: Simulation environment
        num_steps: Number of steps to run (None for infinite)
    """
    # Create visualization
    fig, ax, title, status = create_visualization(env)
    
    # Create scatter plots for each team
    scatter_plots = []
    for team_id, color in enumerate(env.config.TEAM_COLORS):
        team_agents = [a for a in env.agents if a.team_id == team_id and a.is_active]
        if team_agents:
            positions = np.array([a.position for a in team_agents])
            scatter = ax.scatter(positions[:, 0], positions[:, 1],
                               c=color, s=env.config.AGENT_RADIUS*2)
            scatter_plots.append(scatter)
    
    # Create animation
    anim = FuncAnimation(
        fig, update_visualization,
        fargs=(env, ax, title, status, scatter_plots),
        frames=num_steps if num_steps else None,
        interval=1000/env.config.FRAME_RATE,
        blit=True
    )
    
    plt.show()

def main():
    """Main entry point for the simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run swarm simulation')
    parser.add_argument('--steps', type=int, help='Number of steps to run')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig.default()
    if args.config:
        # TODO: Load config from file
        pass
    
    # Override config with command line arguments
    if args.no_vis:
        config.VISUALIZE = False
    
    # Create and run simulation
    env = SwarmEnv(config)
    if config.VISUALIZE:
        run_simulation(env, args.steps)
    else:
        # Run without visualization
        for _ in range(args.steps or 1000):
            env.step()

if __name__ == '__main__':
    main() 