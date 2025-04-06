"""Main simulation runner with visualization."""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional

from .environment.swarm_env import SwarmEnv
from .controllers.random_controller import RandomController
from .config import config

class SwarmSimulation:
    """Main simulation class handling visualization and control.
    
    Attributes:
        env (SwarmEnv): The swarm environment
        controllers (Dict[int, RandomController]): Controllers for each team
        fig (plt.Figure): Matplotlib figure for visualization
        ax (plt.Axes): Matplotlib axes for plotting
        scatter_plots (Dict[int, plt.scatter]): Scatter plots for each team
    """
    
    def __init__(self, env: SwarmEnv):
        """Initialize the simulation.
        
        Args:
            env (SwarmEnv): The swarm environment to simulate
        """
        self.env = env
        self.controllers = {
            team_id: RandomController() 
            for team_id in range(config.NUM_TEAMS)
        }
        
        # Setup visualization
        if config.VISUALIZE:
            plt.style.use('dark_background')
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(0, config.WORLD_SIZE)
            self.ax.set_ylim(0, config.WORLD_SIZE)
            self.ax.set_aspect('equal')
            self.ax.set_title('Swarm vs Swarm Simulation')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            
            # Create scatter plots for each team
            self.scatter_plots = {}
            for team_id in range(config.NUM_TEAMS):
                self.scatter_plots[team_id] = self.ax.scatter(
                    [], [], 
                    c=config.TEAM_COLORS[team_id % len(config.TEAM_COLORS)],
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
        
        self.env.step()
        step_time = time.time() - start_time
        
        # Update visualization
        artists = []
        active_counts = []
        for team_id, scatter in self.scatter_plots.items():
            team_agents = self.env.get_team_agents(team_id)
            active_agents = [agent for agent in team_agents if agent.is_active]
            active_counts.append(len(active_agents))
            
            if active_agents:
                positions = np.array([agent.position for agent in active_agents])
                scatter.set_offsets(positions)
            else:
                scatter.set_offsets(np.zeros((0, 2)))
            artists.append(scatter)
        
        # Update status text
        status = f'Step: {self.env.step_count}\n'
        for team_id in range(config.NUM_TEAMS):
            status += f'Team {team_id}: {active_counts[team_id]} active\n'
        status += f'Step time: {step_time*1000:.1f}ms'
        self.status_text.set_text(status)
        artists.append(self.status_text)
        
        return artists
    
    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the simulation.
        
        Args:
            num_steps (Optional[int]): Number of steps to run, or None for infinite
        """
        if config.VISUALIZE:
            interval = 1000 / config.FRAME_RATE  # Convert FPS to milliseconds
            anim = FuncAnimation(
                self.fig, 
                self.update_visualization,
                frames=num_steps,
                interval=interval,
                blit=True
            )
            plt.show()
        else:
            steps = num_steps if num_steps is not None else config.MAX_STEPS
            for _ in range(steps):
                for team_id, controller in self.controllers.items():
                    team_agents = self.env.get_team_agents(team_id)
                    controller.control_team(team_agents)
                self.env.step()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run swarm simulation')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--steps', type=int, help='Number of steps to run')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    if args.no_vis:
        config.VISUALIZE = False
    
    # Initialize environment
    env = SwarmEnv()
    
    # Create and run simulation
    sim = SwarmSimulation(env)
    sim.run(args.steps)

if __name__ == '__main__':
    main() 