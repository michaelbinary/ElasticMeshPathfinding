import numpy as np
from typing import Dict, List, Tuple
import random
from datetime import datetime
from analytics.metrics import MeshAnalytics
from core.drone_state import DroneState


class ElasticMesh:
    """
    Core simulation class for elastic mesh network of drones with enhanced physics and analytics.
    """

    def __init__(self, space_size: Tuple[float, float, float]):
        # Core properties
        self.space_size = space_size
        self.drones: Dict[str, DroneState] = {}
        self.obstacles: List[np.ndarray] = []
        self.weather_grid = self._initialize_weather_grid()

        # Analytics
        self.analytics = MeshAnalytics()

        # Physics parameters (can be tuned)
        self.physics_params = {
            'k_spring': 0.5,  # Spring constant
            'k_goal': 0.3,  # Goal attraction constant
            'k_repulsion': 0.4,  # Obstacle repulsion constant
            'damping': 0.3,  # Velocity damping
            'min_separation': 2.0,  # Minimum separation between drones
            'max_connection_distance': 8.0,  # Maximum connection range
            'obstacle_influence_radius': 5.0,  # Obstacle effect range
            'weather_influence_factor': 0.2  # Weather effect multiplier
        }

        # Initialize simulation
        self._generate_obstacles()
        self.simulation_start_time = datetime.now()

    def _initialize_weather_grid(self) -> np.ndarray:
        """Initialize 3D weather influence grid with improved patterns."""
        grid_size = (10, 10, 5)
        weather = np.zeros(grid_size)

        # Create more realistic weather patterns using multiple frequencies
        frequencies = [(1 / 5, 1 / 4, 1 / 3), (1 / 3, 1 / 7, 1 / 4), (1 / 4, 1 / 6, 1 / 5)]
        amplitudes = [0.5, 0.3, 0.2]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    # Combine multiple wave patterns
                    weather[i, j, k] = sum(
                        amp * (np.sin(i * fx) + np.cos(j * fy) + np.sin(k * fz))
                        for (fx, fy, fz), amp in zip(frequencies, amplitudes)
                    )

        # Normalize weather values to [-1, 1] range
        weather = weather / np.max(np.abs(weather))
        return weather

    def _generate_obstacles(self):
        """Generate obstacles with improved distribution."""
        num_obstacles = 5
        min_distance = self.physics_params['min_separation'] * 2

        for _ in range(num_obstacles):
            while True:
                position = np.array([
                    random.uniform(0, self.space_size[0]),
                    random.uniform(0, self.space_size[1]),
                    random.uniform(0, self.space_size[2])
                ])

                # Check minimum distance from other obstacles
                if not self.obstacles or all(
                        np.linalg.norm(position - obs) >= min_distance
                        for obs in self.obstacles
                ):
                    self.obstacles.append(position)
                    break

    def add_drone(self, drone_id: str, start: np.ndarray, goal: np.ndarray):
        """Add a drone to the mesh system with validation."""
        # Validate positions
        if not all(0 <= s <= m for s, m in zip(start, self.space_size)):
            raise ValueError("Start position must be within space bounds")
        if not all(0 <= g <= m for g, m in zip(goal, self.space_size)):
            raise ValueError("Goal position must be within space bounds")

        # Initialize drone state
        self.drones[drone_id] = DroneState(
            position=start.astype(float),
            velocity=np.zeros(3),
            goal=goal.astype(float),
            connections=[],
            energy=100.0,
            weather_influence=1.0
        )

        self._update_connections(drone_id)

    def _calculate_spring_force(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """Calculate spring force between two points with improved physics."""
        direction = pos2 - pos1
        distance = np.linalg.norm(direction)

        if distance < 1e-6:  # Prevent division by zero
            return np.zeros(3)

        # Natural length is the minimum separation distance
        deviation = distance - self.physics_params['min_separation']
        force_magnitude = self.physics_params['k_spring'] * deviation

        # Add non-linear effects for extreme stretching
        if distance > self.physics_params['max_connection_distance'] * 0.8:
            force_magnitude *= 1.5  # Increase force when near breaking point

        return force_magnitude * direction / distance

    def _calculate_obstacle_force(self, position: np.ndarray) -> np.ndarray:
        """Calculate repulsive force from obstacles with smooth falloff."""
        total_force = np.zeros(3)
        influence_radius = self.physics_params['obstacle_influence_radius']

        for obstacle in self.obstacles:
            direction = position - obstacle
            distance = np.linalg.norm(direction)

            if distance < influence_radius:
                # Smooth force falloff
                force_magnitude = self.physics_params['k_repulsion'] * (
                        1 - (distance / influence_radius) ** 2
                ) / (distance + 1e-6)
                total_force += force_magnitude * direction

        return total_force

    def _get_weather_influence(self, position: np.ndarray) -> float:
        """Get interpolated weather influence at a given position."""
        # Map position to weather grid coordinates
        grid_pos = np.array([
            int((pos / size) * shape) for pos, size, shape in zip(
                position, self.space_size, self.weather_grid.shape
            )
        ])

        # Clamp coordinates to grid bounds
        grid_pos = np.clip(grid_pos, 0,
                           [s - 1 for s in self.weather_grid.shape])

        # Get surrounding grid points for interpolation
        indices = []
        weights = []
        for i in range(8):  # 8 corners of the surrounding cube
            offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1])
            idx = np.minimum(grid_pos + offset,
                             [s - 1 for s in self.weather_grid.shape])
            indices.append(tuple(idx))

            # Calculate interpolation weight
            weight = 1 - np.linalg.norm(
                (grid_pos + offset - idx) / self.weather_grid.shape
            )
            weights.append(max(0, weight))

        # Normalize weights
        weights = np.array(weights)
        weights /= weights.sum() + 1e-10

        # Interpolate weather values
        return sum(
            self.weather_grid[idx] * w for idx, w in zip(indices, weights)
        )

    def _update_connections(self, drone_id: str):
        """Update mesh connections with improved logic."""
        drone = self.drones[drone_id]
        old_connections = set(drone.connections)
        new_connections = []

        for other_id, other_drone in self.drones.items():
            if other_id != drone_id:
                dist = np.linalg.norm(drone.position - other_drone.position)
                if dist < self.physics_params['max_connection_distance']:
                    new_connections.append(other_id)

        drone.connections = new_connections

        # Track connection changes for analytics
        drone.connection_history.append(len(new_connections))

    def update(self, dt: float):
        """Update the entire mesh system with enhanced physics and analytics."""
        # Calculate forces and update velocities
        new_velocities = {}
        system_energy = 0.0

        for drone_id, drone in self.drones.items():
            # Initialize forces
            total_force = np.zeros(3)

            # Spring forces from connections
            for connected_id in drone.connections:
                connected_drone = self.drones[connected_id]
                spring_force = self._calculate_spring_force(
                    drone.position, connected_drone.position
                )
                total_force += spring_force

                # Add to system energy
                system_energy += 0.5 * self.physics_params['k_spring'] * \
                                 np.linalg.norm(spring_force) ** 2

            # Goal attraction force
            to_goal = drone.goal - drone.position
            distance_to_goal = np.linalg.norm(to_goal)
            if distance_to_goal > 0:
                goal_force = self.physics_params['k_goal'] * \
                             to_goal / distance_to_goal
                total_force += goal_force
                system_energy += self.physics_params['k_goal'] * distance_to_goal

            # Obstacle repulsion
            obstacle_force = self._calculate_obstacle_force(drone.position)
            total_force += obstacle_force

            # Weather influence
            weather_factor = self._get_weather_influence(drone.position)
            total_force *= (1 + self.physics_params['weather_influence_factor'] *
                            weather_factor)

            # Update velocity with damping
            new_velocity = drone.velocity + total_force * dt
            new_velocity *= (1 - self.physics_params['damping'] * dt)
            new_velocities[drone_id] = new_velocity

            # Update drone analytics
            drone.update_analytics(dt)

            # Update energy based on movement and weather
            energy_cost = (np.linalg.norm(new_velocity) * dt * 0.1 *
                           (1 + abs(weather_factor) * 0.2))
            drone.energy = max(0, drone.energy - energy_cost)

        # Update positions and connections
        for drone_id, drone in self.drones.items():
            drone.velocity = new_velocities[drone_id]
            drone.position += drone.velocity * dt

            # Bound positions to space limits
            drone.position = np.clip(
                drone.position, 0,
                [self.space_size[0], self.space_size[1], self.space_size[2]]
            )

            self._update_connections(drone_id)

        # Update analytics
        self.analytics.update_metrics(self.drones, system_energy, self.weather_grid)

    def get_system_state(self) -> dict:
        """Get current state of the entire system."""
        return {
            'time': (datetime.now() - self.simulation_start_time).total_seconds(),
            'drones': {
                drone_id: drone.get_analytics_summary()
                for drone_id, drone in self.drones.items()
            },
            'weather_grid': self.weather_grid.copy(),
            'obstacles': [obs.copy() for obs in self.obstacles]
        }