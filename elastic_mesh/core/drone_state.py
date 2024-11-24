from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from datetime import datetime


@dataclass
class DroneState:
    """
    Represents the state of a drone in the elastic mesh network.

    Attributes:
        position (np.ndarray): 3D position vector
        velocity (np.ndarray): 3D velocity vector
        goal (np.ndarray): 3D goal position
        connections (List[str]): Connected drone IDs
        energy (float): Current energy level (0-100)
        weather_influence (float): Weather effect multiplier (0-1)

    Analytics Attributes:
        distance_traveled (float): Total distance traveled
        time_to_goal (float): Time spent moving towards goal
        avg_connection_count (float): Average number of connections
        connection_history (List[int]): History of connection counts
        creation_time (datetime): When the drone was created
    """
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    connections: List[str]
    energy: float
    weather_influence: float

    # Analytics attributes with defaults
    distance_traveled: float = 0.0
    time_to_goal: float = 0.0
    avg_connection_count: float = 0.0
    connection_history: List[int] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate and initialize computed attributes."""
        # Validate energy bounds
        if not 0 <= self.energy <= 100:
            raise ValueError("Energy must be between 0 and 100")

        # Validate weather influence bounds
        if not 0 <= self.weather_influence <= 1:
            raise ValueError("Weather influence must be between 0 and 1")

        # Ensure numpy arrays are float type
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)

        # Initialize analytics if needed
        if self.connection_history is None:
            self.connection_history = []

        # Store initial distance to goal for progress calculation
        self._initial_distance_to_goal = np.linalg.norm(self.goal - self.position)

    def update_analytics(self, dt: float):
        """Update analytics data for the drone."""
        # Update time tracking
        self.time_to_goal += dt

        # Update connection statistics
        current_connections = len(self.connections)
        self.connection_history.append(current_connections)
        self.avg_connection_count = np.mean(self.connection_history)

        # Calculate distance traveled in this time step
        distance_step = np.linalg.norm(self.velocity) * dt
        self.distance_traveled += distance_step

    def calculate_goal_progress(self) -> float:
        """
        Calculate progress towards goal as a percentage.

        Returns:
            float: Percentage of progress towards goal (0-100)
        """
        current_distance = np.linalg.norm(self.goal - self.position)
        if self._initial_distance_to_goal == 0:
            return 100.0

        progress = (1 - current_distance / self._initial_distance_to_goal) * 100
        return max(0.0, min(100.0, progress))

    def get_analytics_summary(self) -> dict:
        """Get a summary of drone analytics."""
        return {
            'total_distance': self.distance_traveled,
            'time_active': self.time_to_goal,
            'avg_connections': self.avg_connection_count,
            'current_energy': self.energy,
            'uptime': (datetime.now() - self.creation_time).total_seconds(),
            'connection_stability': np.std(self.connection_history) if self.connection_history else 0.0
        }

    def estimate_completion_time(self) -> Optional[float]:
        """
        Estimate time to reach goal based on current velocity and distance.

        Returns:
            Optional[float]: Estimated time in seconds, None if unable to estimate
        """
        if np.all(self.velocity == 0):
            return None

        distance_to_goal = np.linalg.norm(self.goal - self.position)
        current_speed = np.linalg.norm(self.velocity)

        if current_speed > 0:
            return distance_to_goal / current_speed
        return None