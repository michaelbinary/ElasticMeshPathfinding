import pytest
import numpy as np
from datetime import datetime
from core.drone_state import DroneState


@pytest.fixture
def basic_drone_state():
    """Create a basic drone state for testing."""
    return DroneState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        goal=np.array([10.0, 0.0, 0.0]),
        connections=[],
        energy=100.0,
        weather_influence=1.0
    )


class TestDroneState:
    def test_initialization(self, basic_drone_state):
        """Test drone state initialization."""
        assert np.array_equal(basic_drone_state.position, np.array([0.0, 0.0, 0.0]))
        assert np.array_equal(basic_drone_state.velocity, np.array([1.0, 0.0, 0.0]))
        assert basic_drone_state.energy == 100.0
        assert isinstance(basic_drone_state.connection_history, list)
        assert basic_drone_state.creation_time <= datetime.now()
        assert hasattr(basic_drone_state, '_initial_distance_to_goal')

    def test_update_analytics(self, basic_drone_state):
        """Test analytics updates."""
        dt = 0.1
        basic_drone_state.update_analytics(dt)

        # Check distance calculation
        expected_distance = np.linalg.norm(basic_drone_state.velocity) * dt
        assert basic_drone_state.distance_traveled == pytest.approx(expected_distance)

        # Check time tracking
        assert basic_drone_state.time_to_goal == dt

        # Check connection history
        assert len(basic_drone_state.connection_history) == 1
        assert basic_drone_state.connection_history[-1] == 0

    def test_get_analytics_summary(self, basic_drone_state):
        """Test analytics summary generation."""
        summary = basic_drone_state.get_analytics_summary()

        assert 'total_distance' in summary
        assert 'time_active' in summary
        assert 'avg_connections' in summary
        assert 'current_energy' in summary
        assert 'uptime' in summary
        assert 'connection_stability' in summary

    def test_calculate_goal_progress(self):
        """Test goal progress calculation."""
        # Create drone at start position
        drone = DroneState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            goal=np.array([10.0, 0.0, 0.0]),
            connections=[],
            energy=100.0,
            weather_influence=1.0
        )
        assert drone.calculate_goal_progress() == pytest.approx(0.0)

        # Move drone halfway
        drone.position = np.array([5.0, 0.0, 0.0])
        assert drone.calculate_goal_progress() == pytest.approx(50.0)

        # Move to goal
        drone.position = np.array([10.0, 0.0, 0.0])
        assert drone.calculate_goal_progress() == pytest.approx(100.0)

    def test_estimate_completion_time(self, basic_drone_state):
        """Test completion time estimation."""
        # With constant velocity
        estimated_time = basic_drone_state.estimate_completion_time()
        expected_time = 10.0  # Distance = 10, velocity = 1
        assert estimated_time == pytest.approx(expected_time)

        # With zero velocity
        basic_drone_state.velocity = np.zeros(3)
        assert basic_drone_state.estimate_completion_time() is None

    def test_energy_bounds(self):
        """Test energy level bounds."""
        # Test over maximum
        with pytest.raises(ValueError, match="Energy must be between 0 and 100"):
            DroneState(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                goal=np.array([1.0, 0.0, 0.0]),
                connections=[],
                energy=101.0,
                weather_influence=1.0
            )

        # Test under minimum
        with pytest.raises(ValueError, match="Energy must be between 0 and 100"):
            DroneState(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                goal=np.array([1.0, 0.0, 0.0]),
                connections=[],
                energy=-1.0,
                weather_influence=1.0
            )

    def test_connection_history_initialization(self):
        """Test connection history initialization."""
        drone = DroneState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            goal=np.array([1.0, 0.0, 0.0]),
            connections=['drone_2'],
            energy=100.0,
            weather_influence=1.0
        )

        assert isinstance(drone.connection_history, list)
        assert len(drone.connection_history) == 0

    def test_weather_influence_bounds(self):
        """Test weather influence bounds."""
        with pytest.raises(ValueError, match="Weather influence must be between 0 and 1"):
            DroneState(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                goal=np.array([1.0, 0.0, 0.0]),
                connections=[],
                energy=100.0,
                weather_influence=2.0
            )