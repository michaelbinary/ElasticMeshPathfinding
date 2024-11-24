import pytest
import numpy as np
from elastic_mesh.utils.weather import WeatherSystem, WeatherPattern


@pytest.fixture
def weather_system():
    """Create a basic weather system for testing."""
    return WeatherSystem(space_size=(50, 50, 30))


class TestWeatherSystem:
    def test_initialization(self, weather_system):
        """Test weather system initialization."""
        assert weather_system.space_size.tolist() == [50, 50, 30]
        assert weather_system.grid.shape == (10, 10, 5)
        assert len(weather_system.patterns) == 0

    def test_thermal_creation(self, weather_system):
        """Test thermal updraft creation."""
        center = np.array([25, 25, 0])
        weather_system.create_thermal(center, strength=1.2, radius=5.0)

        assert len(weather_system.patterns) == 1
        pattern = weather_system.patterns[0]
        assert pattern.pattern_type == 'thermal'
        assert np.array_equal(pattern.center, center)
        assert pattern.intensity == 1.2
        assert pattern.radius == 5.0

    def test_wind_current(self, weather_system):
        """Test wind current creation."""
        start = np.array([0, 25, 15])
        direction = np.array([1, 0, 0])
        weather_system.create_wind_current(
            start=start,
            direction=direction,
            strength=0.8,
            width=5.0
        )

        assert len(weather_system.patterns) == 1
        pattern = weather_system.patterns[0]
        assert pattern.pattern_type == 'wind'
        assert np.array_equal(pattern.center, start)
        assert np.array_equal(pattern.direction, direction)

    def test_turbulence(self, weather_system):
        """Test turbulence pattern creation."""
        center = np.array([35, 35, 20])
        weather_system.create_turbulence(
            center=center,
            intensity=0.9,
            size=3.0
        )

        assert len(weather_system.patterns) == 1
        pattern = weather_system.patterns[0]
        assert pattern.pattern_type == 'turbulence'
        assert np.array_equal(pattern.center, center)

    def test_influence_calculation(self, weather_system):
        """Test weather influence calculation at positions."""
        # Add patterns
        weather_system.create_thermal(
            center=np.array([25, 25, 0]),
            strength=1.0,
            radius=5.0
        )

        # Test influence at different positions
        positions = [
            np.array([25, 25, 0]),  # At thermal center
            np.array([35, 35, 20]),  # Far from thermal
            np.array([25, 25, 10])  # Above thermal
        ]

        for pos in positions:
            magnitude, direction = weather_system.get_influence_at_position(pos)
            assert isinstance(magnitude, float)
            assert isinstance(direction, np.ndarray)
            assert -1 <= magnitude <= 1
            assert len(direction) == 3

    def test_weather_update(self, weather_system):
        """Test weather pattern evolution over time."""
        # Add a wind pattern
        weather_system.create_wind_current(
            start=np.array([0, 25, 15]),
            direction=np.array([1, 0, 0]),
            strength=0.8
        )

        # Store initial state
        initial_grid = weather_system.grid.copy()

        # Update several times
        for _ in range(10):
            weather_system.update(step=1)

        # Check grid changed
        assert not np.array_equal(weather_system.grid, initial_grid)

        # Check bounds maintained
        assert np.all(weather_system.grid >= -1)
        assert np.all(weather_system.grid <= 1)

    def test_multiple_patterns(self, weather_system):
        """Test interaction of multiple weather patterns."""
        # Add multiple patterns
        weather_system.create_thermal(
            center=np.array([25, 25, 0]),
            strength=1.0,
            radius=5.0
        )
        weather_system.create_wind_current(
            start=np.array([0, 25, 15]),
            direction=np.array([1, 0, 0]),
            strength=0.8
        )
        weather_system.create_turbulence(
            center=np.array([35, 35, 20]),
            intensity=0.9,
            size=3.0
        )

        # Test combined influence
        pos = np.array([25, 25, 15])
        magnitude, direction = weather_system.get_influence_at_position(pos)

        assert isinstance(magnitude, float)
        assert isinstance(direction, np.ndarray)
        assert -1 <= magnitude <= 1

    def test_pattern_bounds(self, weather_system):
        """Test weather patterns respect space bounds."""
        # Try to create pattern outside bounds
        with pytest.raises(ValueError):
            weather_system.create_thermal(
                center=np.array([60, 60, 60]),
                strength=1.0,
                radius=5.0
            )

    def test_pattern_evolution(self, weather_system):
        """Test weather pattern evolution behavior."""
        weather_system.create_thermal(
            center=np.array([25, 25, 0]),
            strength=1.0,
            radius=5.0
        )

        # Store initial pattern state
        initial_pattern = weather_system.patterns[0]
        initial_intensity = initial_pattern.intensity

        # Update many times to ensure evolution
        for _ in range(20):
            weather_system.update(step=1)

        # Check pattern evolved
        final_pattern = weather_system.patterns[0]
        assert final_pattern.intensity != initial_intensity

        # Check evolution bounds
        assert 0.5 <= final_pattern.intensity <= 1.5

    def test_grid_interpolation(self, weather_system):
        """Test weather grid interpolation at arbitrary positions."""
        weather_system.create_thermal(
            center=np.array([25, 25, 0]),
            strength=1.0,
            radius=5.0
        )

        # Test several positions including non-grid points
        test_positions = [
            np.array([24.5, 25.3, 0.7]),  # Between grid points
            np.array([25, 25, 0]),  # Exact grid point
            np.array([0, 0, 0]),  # Boundary
            np.array([49.9, 49.9, 29.9])  # Near boundary
        ]

        for pos in test_positions:
            magnitude, direction = weather_system.get_influence_at_position(pos)
            assert isinstance(magnitude, float)
            assert isinstance(direction, np.ndarray)
            assert -1 <= magnitude <= 1
            assert all(-1 <= d <= 1 for d in direction)