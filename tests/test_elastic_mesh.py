import pytest
import numpy as np
from core.elastic_mesh import ElasticMesh


@pytest.fixture
def basic_mesh():
    """Create a basic mesh for testing."""
    return ElasticMesh(space_size=(50, 50, 30))


@pytest.fixture
def populated_mesh():
    """Create a mesh with multiple drones."""
    mesh = ElasticMesh(space_size=(50, 50, 30))

    # Add drones close enough to form initial connections
    drones = [
        ("drone_1", np.array([10, 10, 15]), np.array([40, 40, 15])),
        ("drone_2", np.array([15, 10, 15]), np.array([35, 40, 15])),  # Within connection range of drone_1
        ("drone_3", np.array([40, 40, 15]), np.array([10, 10, 15]))  # Far from initial positions
    ]

    for drone_id, start, goal in drones:
        mesh.add_drone(drone_id, start, goal)

    # Force update connections
    for drone_id in mesh.drones:
        mesh._update_connections(drone_id)

    return mesh


class TestElasticMesh:
    def test_initialization(self, basic_mesh):
        """Test mesh initialization."""
        assert basic_mesh.space_size == (50, 50, 30)
        assert len(basic_mesh.drones) == 0
        assert isinstance(basic_mesh.obstacles, list)
        assert basic_mesh.weather_grid.shape == (10, 10, 5)

    def test_add_drone(self, basic_mesh):
        """Test adding drones to the mesh."""
        start = np.array([10, 10, 10])
        goal = np.array([40, 40, 10])

        basic_mesh.add_drone("drone_1", start, goal)
        assert "drone_1" in basic_mesh.drones

        # Test adding drone with invalid position
        with pytest.raises(ValueError):
            basic_mesh.add_drone(
                "drone_2",
                np.array([60, 60, 60]),  # Outside space bounds
                goal
            )

    def test_update_connections(self, populated_mesh):
        """Test connection updates between drones."""
        # Initial connections - drones should be connected due to proximity
        drone_1 = populated_mesh.drones["drone_1"]
        initial_connections = set(drone_1.connections)
        assert "drone_2" in initial_connections, "Drones should be initially connected"

        # Move drone_1 far from drone_2
        drone_1.position = np.array([40, 40, 15])
        populated_mesh._update_connections("drone_1")

        # After moving far away, the connection should be broken
        final_connections = set(drone_1.connections)
        assert final_connections != initial_connections, "Connections should change after moving"
        assert "drone_2" not in final_connections, "Connection should be broken due to distance"

    def test_connection_distance_limits(self, basic_mesh):
        """Test connection establishment based on distance limits."""
        # Add two drones just within connection range
        max_dist = basic_mesh.physics_params['max_connection_distance']

        basic_mesh.add_drone(
            "drone_1",
            np.array([10, 10, 15]),
            np.array([40, 40, 15])
        )

        basic_mesh.add_drone(
            "drone_2",
            np.array([10 + max_dist - 0.1, 10, 15]),  # Just within range
            np.array([40, 40, 15])
        )

        basic_mesh._update_connections("drone_1")
        assert "drone_2" in basic_mesh.drones["drone_1"].connections

        # Move second drone just out of range
        basic_mesh.drones["drone_2"].position = np.array([10 + max_dist + 0.1, 10, 15])
        basic_mesh._update_connections("drone_1")
        assert "drone_2" not in basic_mesh.drones["drone_1"].connections

    def test_physics_calculations(self, populated_mesh):
        """Test physics calculations."""
        # Test spring force
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([3.0, 0.0, 0.0])
        force = populated_mesh._calculate_spring_force(pos1, pos2)
        assert isinstance(force, np.ndarray)
        assert force[0] > 0  # Force should pull in positive x direction

        # Test obstacle force
        pos = np.array([0.0, 0.0, 0.0])
        force = populated_mesh._calculate_obstacle_force(pos)
        assert isinstance(force, np.ndarray)
        assert len(force) == 3

        # Test weather influence
        influence = populated_mesh._get_weather_influence(pos)
        assert isinstance(influence, float)
        assert -1 <= influence <= 1

    def test_system_update(self, populated_mesh):
        """Test full system update."""
        # Store initial positions
        initial_positions = {
            drone_id: drone.position.copy()
            for drone_id, drone in populated_mesh.drones.items()
        }

        # Update system
        populated_mesh.update(0.1)

        # Check positions changed
        for drone_id, drone in populated_mesh.drones.items():
            assert not np.array_equal(drone.position, initial_positions[drone_id])

    def test_energy_consumption(self, populated_mesh):
        """Test energy consumption during movement."""
        # Store initial energy levels
        initial_energy = {
            drone_id: drone.energy
            for drone_id, drone in populated_mesh.drones.items()
        }

        # Run several updates
        for _ in range(10):
            populated_mesh.update(0.1)

        # Check energy decreased
        for drone_id, drone in populated_mesh.drones.items():
            assert drone.energy < initial_energy[drone_id]

    def test_obstacle_generation(self, basic_mesh):
        """Test obstacle generation."""
        # Check obstacles were generated
        assert len(basic_mesh.obstacles) > 0

        # Check obstacles are within bounds
        for obstacle in basic_mesh.obstacles:
            assert all(0 <= coord <= size
                       for coord, size in zip(obstacle, basic_mesh.space_size))

    def test_weather_system(self, basic_mesh):
        """Test weather system initialization and updates."""
        # Check weather grid initialization
        assert basic_mesh.weather_grid is not None
        assert isinstance(basic_mesh.weather_grid, np.ndarray)

        # Check weather influence calculation
        pos = np.array([25, 25, 15])
        influence = basic_mesh._get_weather_influence(pos)
        assert isinstance(influence, float)
        assert -1 <= influence <= 1

    def test_system_state(self, populated_mesh):
        """Test system state retrieval."""
        state = populated_mesh.get_system_state()

        assert 'time' in state
        assert 'drones' in state
        assert 'weather_grid' in state
        assert 'obstacles' in state

        # Check drone state
        for drone_id, drone_state in state['drones'].items():
            assert 'total_distance' in drone_state
            assert 'avg_connections' in drone_state
            assert 'current_energy' in drone_state

    def test_boundary_conditions(self, basic_mesh):
        """Test handling of boundary conditions."""
        # Add drone near boundary
        start = np.array([48, 48, 28])
        goal = np.array([49, 49, 29])
        basic_mesh.add_drone("boundary_drone", start, goal)

        # Update system
        basic_mesh.update(0.1)

        # Check drone stays within bounds
        drone = basic_mesh.drones["boundary_drone"]
        assert all(0 <= coord <= size
                   for coord, size in zip(drone.position, basic_mesh.space_size))