from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
from core.drone_state import DroneState


class MeshAnalytics:
    """
    Handles analytics and metrics collection for the elastic mesh system.
    """

    def __init__(self):
        self.metrics_history = {
            'timestamp': [],
            'system_energy': [],
            'mesh_density': [],
            'network_efficiency': [],
            'avg_drone_energy': [],
            'total_distance': [],
            'avg_connections': [],
            'weather_impact': []
        }

        self.start_time = datetime.now()

    def update_metrics(self, drones: Dict[str, DroneState], system_energy: float,
                       weather_grid: np.ndarray):
        """
        Update analytics metrics for the current time step.

        Args:
            drones: Dictionary of all drones in the system
            system_energy: Current system energy level
            weather_grid: Current weather influence grid
        """
        timestamp = datetime.now()

        # Basic metrics
        avg_energy = np.mean([drone.energy for drone in drones.values()])
        total_distance = sum(drone.distance_traveled for drone in drones.values())
        avg_connections = np.mean([len(drone.connections) for drone in drones.values()])

        # Calculate mesh density
        total_possible = len(drones) * (len(drones) - 1) / 2
        actual_connections = sum(len(drone.connections) for drone in drones.values()) / 2
        mesh_density = actual_connections / total_possible if total_possible > 0 else 0

        # Calculate network efficiency
        network_efficiency = self._calculate_network_efficiency(drones)

        # Calculate weather impact
        weather_impact = np.mean(np.abs(weather_grid))

        # Store metrics
        self.metrics_history['timestamp'].append(timestamp)
        self.metrics_history['system_energy'].append(system_energy)
        self.metrics_history['mesh_density'].append(mesh_density)
        self.metrics_history['network_efficiency'].append(network_efficiency)
        self.metrics_history['avg_drone_energy'].append(avg_energy)
        self.metrics_history['total_distance'].append(total_distance)
        self.metrics_history['avg_connections'].append(avg_connections)
        self.metrics_history['weather_impact'].append(weather_impact)

    def _calculate_network_efficiency(self, drones: Dict[str, DroneState]) -> float:
        """Calculate network efficiency using NetworkX."""
        if len(drones) < 2:
            return 0.0

        G = nx.Graph()

        # Add all drones as nodes
        for drone_id in drones:
            G.add_node(drone_id)

        # Add connections as edges
        for drone_id, drone in drones.items():
            for connected_id in drone.connections:
                G.add_edge(drone_id, connected_id)

        # Calculate efficiency
        if not nx.is_connected(G):
            return 0.0

        return nx.global_efficiency(G)

    def generate_report(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a comprehensive analytics report.

        Args:
            save_path: Optional path to save the report as CSV

        Returns:
            pd.DataFrame: Analytics data
        """
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics_history)

        # Calculate summary statistics
        summary = {
            'simulation_duration': (datetime.now() - self.start_time).total_seconds(),
            'avg_system_energy': np.mean(self.metrics_history['system_energy']),
            'peak_network_efficiency': max(self.metrics_history['network_efficiency']),
            'avg_mesh_density': np.mean(self.metrics_history['mesh_density']),
            'total_distance_covered': self.metrics_history['total_distance'][-1],
            'final_avg_energy': self.metrics_history['avg_drone_energy'][-1],
            'avg_weather_impact': np.mean(self.metrics_history['weather_impact'])
        }

        # Add summary to DataFrame
        df.attrs['summary'] = summary

        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)

        return df

    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Generate plots for key metrics.

        Args:
            save_path: Optional path to save the plots
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # System Energy
        axes[0, 0].plot(self.metrics_history['system_energy'])
        axes[0, 0].set_title('System Energy Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Energy')

        # Network Efficiency
        axes[0, 1].plot(self.metrics_history['network_efficiency'])
        axes[0, 1].set_title('Network Efficiency')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Efficiency')

        # Mesh Density
        axes[1, 0].plot(self.metrics_history['mesh_density'])
        axes[1, 0].set_title('Mesh Density')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Density')

        # Average Drone Energy
        axes[1, 1].plot(self.metrics_history['avg_drone_energy'])
        axes[1, 1].set_title('Average Drone Energy')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Energy Level (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()