import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
import colorsys
from typing import Dict, Optional, List
from rich.console import Console
from utils.visualization_helpers import VisualizationHelper
from utils.weather import WeatherSystem

console = Console()


class MeshVisualizer:
    """
    Handles 3D visualization of the elastic mesh simulation with enhanced graphics
    and animation capabilities.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.space_size = mesh.space_size
        self.frames: List[np.ndarray] = []

        # Initialize weather system if not present
        if not hasattr(self.mesh, 'weather_system'):
            self.mesh.weather_system = WeatherSystem(self.space_size)

        # Color scheme
        self.drone_colors = self._generate_drone_colors()
        self.setup_visualization()

        # Animation settings
        self.animation_settings = {
            'fps': 30,
            'quality': 95,
            'dpi': 150,
            'duration': 15  # seconds
        }

    def _generate_drone_colors(self) -> Dict[str, tuple]:
        """Generate visually distinct colors for drones."""
        colors = {}
        for i, drone_id in enumerate(self.mesh.drones.keys()):
            hue = i / max(len(self.mesh.drones), 1)
            colors[drone_id] = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return colors

    def setup_visualization(self):
        """Configure the 3D visualization environment."""
        self.ax.set_xlim(0, self.space_size[0])
        self.ax.set_ylim(0, self.space_size[1])
        self.ax.set_zlim(0, self.space_size[2])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Improve 3D view
        self.ax.view_init(elev=20, azim=45)
        self.ax.grid(True, alpha=0.3)

    def capture_frame(self, fig):
        """Capture current figure as a frame."""
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def add_telemetry_overlay(self, position: str = 'right'):
        """
        Add telemetry information overlay to the visualization.

        Args:
            position (str): Position of overlay ('right' or 'bottom')
        """
        # Create overlay axes
        if position == 'right':
            ax_overlay = self.fig.add_axes([0.85, 0.1, 0.15, 0.8])
        else:  # 'bottom'
            ax_overlay = self.fig.add_axes([0.1, 0.02, 0.8, 0.1])

        ax_overlay.axis('off')

        # Compile telemetry text
        text_lines = [
            "System Telemetry",
            "",
            f"Active Drones: {len(self.mesh.drones)}",
            f"Time Step: {len(self.frames)}"
        ]

        # Add drone-specific information
        for drone_id, drone in self.mesh.drones.items():
            text_lines.extend([
                f"\n{drone_id}:",
                f"  Energy: {drone.energy:.1f}%",
                f"  Connections: {len(drone.connections)}"
            ])

        # Add analytics if available
        if hasattr(self.mesh, 'analytics'):
            metrics = self.mesh.analytics.metrics_history
            if metrics.get('network_efficiency', []):
                text_lines.append(
                    f"\nNetwork Efficiency: {metrics['network_efficiency'][-1]:.2f}"
                )
            if metrics.get('mesh_density', []):
                text_lines.append(
                    f"Mesh Density: {metrics['mesh_density'][-1]:.2f}"
                )

        # Join text lines
        text_content = '\n'.join(text_lines)

        # Add text to overlay
        text = ax_overlay.text(
            0.05, 0.95,  # Position within overlay
            text_content,
            transform=ax_overlay.transAxes,
            fontsize=8,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='none',
                boxstyle='round,pad=0.5'
            )
        )

        self.fig.canvas.draw()
        return text

    def update(self, frame):
        """Update visualization for current frame."""
        self.ax.clear()
        self.setup_visualization()

        # Draw weather influence
        self._draw_weather()

        # Draw obstacles
        self._draw_obstacles()

        # Draw drones and connections
        self._draw_drones()

        # Update physics
        self.mesh.update(0.1)

        # Capture frame
        self.capture_frame(self.fig)

        return self.ax,

    def _draw_weather(self):
        """Draw weather patterns."""
        x, y, z = np.mgrid[0:self.space_size[0]:10j,
                  0:self.space_size[1]:10j,
                  0:self.space_size[2]:10j]

        weather_data = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(y.shape[1]):
                for k in range(z.shape[2]):
                    pos = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                    if hasattr(self.mesh, 'weather_system'):
                        magnitude, _ = self.mesh.weather_system.get_influence_at_position(pos)
                        weather_data[i, j, k] = magnitude

        scatter = self.ax.scatter(x, y, z,
                                  c=weather_data,
                                  alpha=0.1,
                                  cmap='Blues')
        return scatter

    def _draw_obstacles(self):
        """Draw obstacles and their influence zones."""
        for obstacle in self.mesh.obstacles:
            VisualizationHelper.draw_obstacle(
                self.ax,
                obstacle,
                self.mesh.physics_params['obstacle_influence_radius']
            )

    def _draw_drones(self):
        """Draw drones and their connections."""
        for drone_id, drone in self.mesh.drones.items():
            color = self.drone_colors[drone_id]

            # Draw drone
            self.ax.scatter(*drone.position,
                            color=color,
                            s=100,
                            label=f'{drone_id} ({drone.energy:.1f}%)')

            # Draw connections
            for connected_id in drone.connections:
                connected_drone = self.mesh.drones[connected_id]
                VisualizationHelper.draw_connection_strength(
                    self.ax,
                    drone.position,
                    connected_drone.position,
                    1.0  # Connection strength (could be calculated based on distance)
                )

    def create_animation(self, duration: Optional[float] = None, save_path: Optional[str] = None):
        """Create and optionally save animation."""
        console.print("[bold blue]Creating animation...[/bold blue]")

        if duration is not None:
            self.animation_settings['duration'] = duration

        num_frames = int(self.animation_settings['fps'] *
                         self.animation_settings['duration'])

        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=num_frames,
            interval=1000 / self.animation_settings['fps'],
            blit=False,
            repeat=False
        )

        if save_path:
            console.print("[yellow]Saving animation...[/yellow]")
            writer = PillowWriter(
                fps=self.animation_settings['fps'],
                metadata=dict(artist='ElasticMesh'),
                bitrate=self.animation_settings['bitrate']
            )

            anim.save(
                save_path,
                writer=writer,
                dpi=self.animation_settings['dpi']
            )

            console.print(f"[green]Animation saved to {save_path}[/green]")

        return anim

    def save_frame(self, path: str):
        """Save current visualization frame."""
        self.fig.savefig(path,
                         dpi=self.animation_settings['dpi'],
                         bbox_inches='tight')
        console.print(f"Frame saved to {path}")

    def show(self):
        """Display the current visualization."""
        plt.show()