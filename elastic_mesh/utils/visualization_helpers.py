import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Optional
import colorsys


class VisualizationHelper:
    """Helper class for enhanced visualization features."""

    @staticmethod
    def create_custom_colormap(name: str = 'mesh_colors') -> LinearSegmentedColormap:
        """Create a custom colormap for the mesh visualization."""
        colors = [
            (0.0, '#1f77b4'),  # Blue
            (0.3, '#2ca02c'),  # Green
            (0.6, '#ff7f0e'),  # Orange
            (1.0, '#d62728')  # Red
        ]
        return LinearSegmentedColormap.from_list(name, colors)

    @staticmethod
    def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
        """Generate visually distinct colors for multiple drones."""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.8 + np.random.normal(0, 0.1)
            value = 0.9 + np.random.normal(0, 0.1)

            # Clamp values
            saturation = np.clip(saturation, 0.6, 1.0)
            value = np.clip(value, 0.7, 1.0)

            colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
        return colors

    @staticmethod
    def setup_3d_axes(ax: Axes3D,
                      space_size: Tuple[float, float, float],
                      title: Optional[str] = None):
        """Configure 3D axes with improved styling."""
        # Set limits
        ax.set_xlim(0, space_size[0])
        ax.set_ylim(0, space_size[1])
        ax.set_zlim(0, space_size[2])

        # Labels
        ax.set_xlabel('X Distance (m)')
        ax.set_ylabel('Y Distance (m)')
        ax.set_zlabel('Z Distance (m)')

        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--')

        # Background and pane colors
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set title if provided
        if title:
            ax.set_title(title, pad=20)

        # Improve 3D view
        ax.view_init(elev=20, azim=45)

    @staticmethod
    def draw_drone_trail(ax: Axes3D,
                         positions: np.ndarray,
                         color: Tuple[float, float, float],
                         fade_factor: float = 0.1):
        """Draw a fading trail behind a drone."""
        num_positions = len(positions)
        if num_positions < 2:
            return

        for i in range(num_positions - 1):
            alpha = np.exp(-fade_factor * (num_positions - i - 1))
            ax.plot3D(
                positions[i:i + 2, 0],
                positions[i:i + 2, 1],
                positions[i:i + 2, 2],
                color=color,
                alpha=alpha,
                linewidth=1
            )

    @staticmethod
    def draw_connection_strength(ax: Axes3D,
                                 pos1: np.ndarray,
                                 pos2: np.ndarray,
                                 strength: float,
                                 max_strength: float = 1.0):
        """Draw a connection line with visual indication of strength."""
        # Calculate connection properties
        distance = np.linalg.norm(pos2 - pos1)
        normalized_strength = strength / max_strength

        # Define connection style based on strength
        if normalized_strength > 0.8:
            linestyle = '-'
            linewidth = 2.0
            alpha = 0.8
        elif normalized_strength > 0.5:
            linestyle = '--'
            linewidth = 1.5
            alpha = 0.6
        else:
            linestyle = ':'
            linewidth = 1.0
            alpha = 0.4

        # Draw connection
        ax.plot3D(
            [pos1[0], pos2[0]],
            [pos1[1], pos2[1]],
            [pos1[2], pos2[2]],
            color='gray',
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha
        )

    @staticmethod
    def draw_energy_indicator(ax: Axes3D,
                              position: np.ndarray,
                              energy: float,
                              size: float = 100):
        """Draw an energy level indicator around a drone."""
        # Create circle parameters
        theta = np.linspace(0, 2 * np.pi, 32)
        radius = size * (energy / 100)

        # Calculate circle points
        x = position[0] + radius * np.cos(theta)
        y = position[1] + radius * np.sin(theta)
        z = np.full_like(theta, position[2])

        # Color based on energy level
        if energy > 75:
            color = '#2ecc71'  # Green
        elif energy > 50:
            color = '#f1c40f'  # Yellow
        elif energy > 25:
            color = '#e67e22'  # Orange
        else:
            color = '#e74c3c'  # Red

        # Draw energy indicator
        ax.plot3D(x, y, z, color=color, alpha=0.3)

    @staticmethod
    def draw_goal_marker(ax: Axes3D,
                         position: np.ndarray,
                         size: float = 100,
                         achieved: bool = False):
        """Draw an attractive goal marker."""
        marker_style = '*' if not achieved else '✓'
        color = '#3498db' if not achieved else '#27ae60'

        # Draw main marker
        ax.scatter(*position, marker=marker_style, s=size * 1.5,
                   color=color, alpha=0.8)

        # Draw pulsing rings if not achieved
        if not achieved:
            for i in range(3):
                theta = np.linspace(0, 2 * np.pi, 32)
                phi = np.linspace(0, np.pi, 16)
                theta, phi = np.meshgrid(theta, phi)

                r = size * (1.2 + i * 0.2)
                x = position[0] + r * np.sin(phi) * np.cos(theta)
                y = position[1] + r * np.sin(phi) * np.sin(theta)
                z = position[2] + r * np.cos(phi)

                ax.plot_surface(x, y, z, color=color, alpha=0.1 / (i + 1))

    @staticmethod
    def draw_velocity_vector(ax: Axes3D,
                             position: np.ndarray,
                             velocity: np.ndarray,
                             scale: float = 1.0):
        """Draw a velocity vector with speed indication."""
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return

        # Normalize and scale velocity
        direction = velocity / speed
        vector_length = np.log1p(speed) * scale

        # Draw main vector
        ax.quiver(*position,
                  *direction,
                  length=vector_length,
                  color='#2980b9',
                  arrow_length_ratio=0.2,
                  alpha=0.6)

    @staticmethod
    def draw_obstacle(ax: Axes3D,
                      position: np.ndarray,
                      influence_radius: float,
                      danger_level: float = 1.0):
        """Draw an obstacle with its influence zone."""
        # Draw obstacle marker
        ax.scatter(*position,
                   marker='^',
                   s=100,
                   color='#c0392b',
                   label='Obstacle')

        # Draw influence sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        x = position[0] + influence_radius * np.outer(np.cos(u), np.sin(v))
        y = position[1] + influence_radius * np.outer(np.sin(u), np.sin(v))
        z = position[2] + influence_radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z,
                        color='#e74c3c',
                        alpha=0.1 * danger_level)

    @staticmethod
    def create_animation_frame(ax: Axes3D,
                               mesh_state: dict,
                               frame_number: int) -> plt.Figure:
        """Create a complete frame for animation."""
        # Clear previous frame
        ax.clear()

        # Setup basic axes
        VisualizationHelper.setup_3d_axes(
            ax,
            mesh_state['space_size'],
            f'Simulation Time: {frame_number / 30:.2f}s'
        )

        # Draw all elements
        for drone_id, drone_data in mesh_state['drones'].items():
            # Draw drone
            ax.scatter(*drone_data['position'],
                       color=drone_data['color'],
                       s=100,
                       label=f'Drone {drone_id}')

            # Draw energy indicator
            VisualizationHelper.draw_energy_indicator(
                ax,
                drone_data['position'],
                drone_data['energy']
            )

            # Draw velocity
            VisualizationHelper.draw_velocity_vector(
                ax,
                drone_data['position'],
                drone_data['velocity']
            )

            # Draw trail
            if 'history' in drone_data:
                VisualizationHelper.draw_drone_trail(
                    ax,
                    drone_data['history'],
                    drone_data['color']
                )

            # Draw connections
            for conn_id in drone_data['connections']:
                conn_data = mesh_state['drones'][conn_id]
                VisualizationHelper.draw_connection_strength(
                    ax,
                    drone_data['position'],
                    conn_data['position'],
                    drone_data['connection_strengths'][conn_id]
                )

            # Draw goal
            VisualizationHelper.draw_goal_marker(
                ax,
                drone_data['goal'],
                achieved=drone_data.get('goal_reached', False)
            )

        # Draw obstacles
        for obstacle in mesh_state['obstacles']:
            VisualizationHelper.draw_obstacle(
                ax,
                obstacle['position'],
                obstacle['influence_radius'],
                obstacle['danger_level']
            )

        # Add weather visualization if present
        if 'weather' in mesh_state:
            weather_scatter = ax.scatter(
                mesh_state['weather']['x'],
                mesh_state['weather']['y'],
                mesh_state['weather']['z'],
                c=mesh_state['weather']['intensity'],
                cmap='coolwarm',
                alpha=0.1
            )
            plt.colorbar(weather_scatter, label='Weather Intensity')

        return ax.figure

    @staticmethod
    def save_frame(fig: plt.Figure,
                   filename: str,
                   dpi: int = 100):
        """Save a single frame to file."""
        fig.savefig(filename,
                    dpi=dpi,
                    bbox_inches='tight',
                    pad_inches=0.1)

    @staticmethod
    def add_telemetry_overlay(fig: plt.Figure,
                              mesh_state: dict,
                              position: str = 'right'):
        """Add telemetry information overlay to the figure."""
        if position == 'right':
            ax = fig.add_axes([0.85, 0.1, 0.15, 0.8])
        else:  # 'bottom'
            ax = fig.add_axes([0.1, 0.02, 0.8, 0.1])

        ax.axis('off')

        # Compile telemetry text
        text = "System Telemetry\n\n"
        text += f"Time: {mesh_state['time']:.2f}s\n"
        text += f"Active Drones: {len(mesh_state['drones'])}\n"
        text += f"Network Efficiency: {mesh_state['network_efficiency']:.2f}\n"
        text += f"System Energy: {mesh_state['system_energy']:.2f}\n"
        text += f"Weather Impact: {mesh_state['weather_impact']:.2f}\n"

        ax.text(0, 1, text,
                fontsize=8,
                verticalalignment='top',
                fontfamily='monospace')