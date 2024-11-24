import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import scipy.ndimage as ndimage
from rich.console import Console

console = Console()


@dataclass
class WeatherPattern:
    """Represents a weather pattern in the simulation space."""
    center: np.ndarray  # Center position of the pattern
    intensity: float  # Strength of the weather effect
    radius: float  # Radius of influence
    pattern_type: str  # Type of weather pattern (e.g., 'turbulence', 'wind', 'thermal')
    direction: Optional[np.ndarray] = None  # Direction for directional patterns


class WeatherSystem:
    """Manages weather patterns and their influence on the simulation space."""

    def __init__(self, space_size: Tuple[float, float, float], resolution: Tuple[int, int, int] = (10, 10, 5)):
        self.space_size = np.array(space_size)
        self.resolution = resolution
        self.grid = np.zeros(resolution)
        self.patterns: List[WeatherPattern] = []

        # Weather generation parameters
        self.turbulence_scale = 0.3
        self.wind_strength = 0.5
        self.update_frequency = 10  # Steps between weather updates
        self.step_counter = 0

    def add_pattern(self, pattern: WeatherPattern):
        """Add a new weather pattern to the system."""
        self.patterns.append(pattern)
        self._update_weather_grid()

    def create_thermal(self, center: np.ndarray, strength: float = 1.0, radius: float = 5.0):
        """Create a thermal updraft pattern."""
        pattern = WeatherPattern(
            center=np.array(center),
            intensity=strength,
            radius=radius,
            pattern_type='thermal',
            direction=np.array([0, 0, 1])  # Upward direction
        )
        self.add_pattern(pattern)

    def create_wind_current(self,
                            start: np.ndarray,
                            direction: np.ndarray,
                            strength: float = 1.0,
                            width: float = 5.0):
        """Create a directional wind current."""
        pattern = WeatherPattern(
            center=np.array(start),
            intensity=strength,
            radius=width,
            pattern_type='wind',
            direction=direction / np.linalg.norm(direction)
        )
        self.add_pattern(pattern)

    def create_turbulence(self, center: np.ndarray, intensity: float = 1.0, size: float = 3.0):
        """Create a turbulent area."""
        pattern = WeatherPattern(
            center=np.array(center),
            intensity=intensity,
            radius=size,
            pattern_type='turbulence'
        )
        self.add_pattern(pattern)

    def _update_weather_grid(self):
        """Update the weather influence grid based on all patterns."""
        self.grid = np.zeros(self.resolution)

        # Create coordinate meshgrid
        x, y, z = np.meshgrid(
            np.linspace(0, self.space_size[0], self.resolution[0]),
            np.linspace(0, self.space_size[1], self.resolution[1]),
            np.linspace(0, self.space_size[2], self.resolution[2]),
            indexing='ij'
        )
        coords = np.stack([x, y, z], axis=-1)

        # Apply each pattern's influence
        for pattern in self.patterns:
            if pattern.pattern_type == 'thermal':
                self._apply_thermal(coords, pattern)
            elif pattern.pattern_type == 'wind':
                self._apply_wind(coords, pattern)
            elif pattern.pattern_type == 'turbulence':
                self._apply_turbulence(coords, pattern)

        # Add some randomness for realism
        self.grid += np.random.normal(0, 0.1, self.grid.shape)

        # Smooth the grid
        self.grid = ndimage.gaussian_filter(self.grid, sigma=1.0)

        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(self.grid))
        if max_val > 0:
            self.grid /= max_val

    def _apply_thermal(self, coords: np.ndarray, pattern: WeatherPattern):
        """Apply thermal updraft pattern to the grid."""
        distances = np.linalg.norm(coords - pattern.center, axis=-1)
        influence = np.exp(-distances ** 2 / (2 * pattern.radius ** 2))

        # Create vertical gradient
        height_factor = coords[..., 2] / self.space_size[2]
        influence *= height_factor

        self.grid += influence * pattern.intensity

    def _apply_wind(self, coords: np.ndarray, pattern: WeatherPattern):
        """Apply wind current pattern to the grid."""
        # Calculate signed distance from wind direction line
        to_point = coords - pattern.center
        along_wind = np.dot(to_point, pattern.direction)
        perpendicular = np.linalg.norm(
            to_point - along_wind[..., np.newaxis] * pattern.direction,
            axis=-1
        )

        # Calculate influence based on distance from wind path
        influence = np.exp(-perpendicular ** 2 / (2 * pattern.radius ** 2))

        # Add directional component
        direction_influence = np.dot(
            np.ones_like(coords), pattern.direction
        )

        self.grid += influence * direction_influence * pattern.intensity

    def _apply_turbulence(self, coords: np.ndarray, pattern: WeatherPattern):
        """Apply turbulent pattern to the grid."""
        distances = np.linalg.norm(coords - pattern.center, axis=-1)
        base_influence = np.exp(-distances ** 2 / (2 * pattern.radius ** 2))

        # Add turbulent noise
        noise = np.random.normal(
            0, self.turbulence_scale,
            size=self.resolution
        )
        smoothed_noise = ndimage.gaussian_filter(noise, sigma=1.0)

        self.grid += base_influence * smoothed_noise * pattern.intensity

    def get_influence_at_position(self, position: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Get weather influence and direction at a specific position.

        Returns:
            Tuple[float, np.ndarray]: Influence magnitude and direction vector
        """
        # Convert position to grid coordinates
        grid_pos = (position / self.space_size *
                    np.array(self.resolution)).astype(int)
        grid_pos = np.clip(grid_pos, 0,
                           np.array(self.resolution) - 1)

        # Get magnitude from grid
        magnitude = self.grid[tuple(grid_pos)]

        # Calculate gradient for direction
        gradient = np.gradient(self.grid)
        direction = np.array([g[tuple(grid_pos)] for g in gradient])
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            direction /= direction_norm

        return magnitude, direction

    def update(self, step: int):
        """Update weather patterns periodically."""
        self.step_counter += 1
        if self.step_counter >= self.update_frequency:
            self.step_counter = 0
            self._update_weather_grid()

            # Add some pattern evolution
            for pattern in self.patterns:
                if pattern.pattern_type == 'wind':
                    # Slightly vary wind direction
                    angle = np.random.normal(0, 0.1)
                    rotation = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])
                    pattern.direction = rotation @ pattern.direction
                elif pattern.pattern_type == 'thermal':
                    # Vary thermal strength
                    pattern.intensity *= (1 + np.random.normal(0, 0.1))
                    pattern.intensity = np.clip(pattern.intensity, 0.5, 1.5)

    def visualize_weather(self, ax, alpha: float = 0.1):
        """Visualize weather patterns in a 3D plot."""
        x, y, z = np.meshgrid(
            np.linspace(0, self.space_size[0], self.resolution[0]),
            np.linspace(0, self.space_size[1], self.resolution[1]),
            np.linspace(0, self.space_size[2], self.resolution[2]),
            indexing='ij'
        )

        # Plot weather influence as scatter points
        scatter = ax.scatter(
            x, y, z,
            c=self.grid,
            cmap='coolwarm',
            alpha=alpha,
            vmin=-1,
            vmax=1
        )

        # Add wind arrows for significant flows
        stride = 2
        quiver_mask = np.abs(self.grid) > 0.3
        masked_positions = np.column_stack([
            x[quiver_mask][::stride],
            y[quiver_mask][::stride],
            z[quiver_mask][::stride]
        ])

        if len(masked_positions) > 0:
            gradients = np.array([g[quiver_mask][::stride]
                                  for g in np.gradient(self.grid)])
            ax.quiver(
                masked_positions[:, 0],
                masked_positions[:, 1],
                masked_positions[:, 2],
                gradients[0],
                gradients[1],
                gradients[2],
                length=2.0,
                normalize=True,
                alpha=0.3
            )

        return scatter