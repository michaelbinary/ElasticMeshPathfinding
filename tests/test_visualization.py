import pytest
import numpy as np
import matplotlib.pyplot as plt
from simulation.visualization import MeshVisualizer
from core.elastic_mesh import ElasticMesh
from utils.visualization_helpers import VisualizationHelper


@pytest.fixture
def basic_visualizer():
    """Create a basic visualizer for testing."""
    mesh = ElasticMesh(space_size=(50, 50, 30))
    return MeshVisualizer(mesh)


@pytest.fixture
def populated_visualizer():
    """Create a visualizer with drones."""
    mesh = ElasticMesh(space_size=(50, 50, 30))

    # Add test drones
    mesh.add_drone(
        "drone_1",
        np.array([10, 10, 15]),
        np.array([40, 40, 15])
    )
    mesh.add_drone(
        "drone_2",
        np.array([15, 10, 15]),
        np.array([35, 40, 15])
    )

    return MeshVisualizer(mesh)


class TestVisualization:
    def test_initialization(self, basic_visualizer):
        """Test visualizer initialization."""
        assert basic_visualizer.fig is not None
        assert basic_visualizer.ax is not None
        assert basic_visualizer.space_size == (50, 50, 30)

    def test_color_generation(self, populated_visualizer):
        """Test drone color generation."""
        colors = populated_visualizer.drone_colors
        assert len(colors) == len(populated_visualizer.mesh.drones)
        assert all(len(color) == 3 for color in colors.values())
        assert all(0 <= c <= 1 for color in colors.values() for c in color)

    def test_frame_capture(self, populated_visualizer):
        """Test frame capture functionality."""
        populated_visualizer.capture_frame(populated_visualizer.fig)
        assert len(populated_visualizer.frames) == 1
        assert isinstance(populated_visualizer.frames[0], np.ndarray)

    def test_animation_creation(self, populated_visualizer):
        """Test animation creation."""
        anim = populated_visualizer.create_animation(
            duration=1,
            save_path=None
        )
        assert anim is not None

    def test_weather_visualization(self, populated_visualizer):
        """Test weather pattern visualization."""
        populated_visualizer.mesh.weather_system.create_thermal(
            center=np.array([25, 25, 0]),
            strength=1.0,
            radius=5.0
        )

        populated_visualizer.update(frame=0)
        # Check weather scatter plot exists
        assert any(isinstance(child, plt.matplotlib.collections.PathCollection)
                   for child in populated_visualizer.ax.get_children())

    def test_drone_visualization(self, populated_visualizer):
        """Test drone visualization elements."""
        populated_visualizer.update(frame=0)

        # Check drone markers exist
        children = populated_visualizer.ax.get_children()
        assert any(isinstance(child, plt.matplotlib.collections.PathCollection)
                   for child in children)

        # Check connection lines exist
        assert any(isinstance(child, plt.matplotlib.lines.Line2D)
                   for child in children)

    def test_telemetry_overlay(self, populated_visualizer):
        """Test telemetry information overlay."""
        # Add some frames to simulate animation progress
        populated_visualizer.frames = [np.zeros((100, 100, 3)) for _ in range(5)]

        # Add the overlay
        text = populated_visualizer.add_telemetry_overlay(position='right')

        # Verify text was added
        assert isinstance(text, plt.Text)

        # Check content
        assert "System Telemetry" in text.get_text()
        assert "Active Drones" in text.get_text()

        # Check position
        assert text.get_position()[0] == 0.05  # x position
        assert text.get_position()[1] == 0.95  # y position

        # Check styling
        assert text.get_fontsize() == 8
        assert text.get_verticalalignment() == 'top'
        assert text.get_fontfamily()[0] == 'monospace'

        # Verify the overlay axes exists
        overlay_axes = [ax for ax in populated_visualizer.fig.axes
                        if ax != populated_visualizer.ax]
        assert len(overlay_axes) == 1
        assert not overlay_axes[0].get_xaxis().get_visible()
        assert not overlay_axes[0].get_yaxis().get_visible()

    def test_save_frame(self, populated_visualizer, tmp_path):
        """Test frame saving functionality."""
        frame_path = tmp_path / "test_frame.png"
        populated_visualizer.save_frame(str(frame_path))
        assert frame_path.exists()

    def test_visualization_bounds(self, populated_visualizer):
        """Test visualization maintains proper bounds."""
        populated_visualizer.update(frame=0)

        xlim = populated_visualizer.ax.get_xlim()
        ylim = populated_visualizer.ax.get_ylim()
        zlim = populated_visualizer.ax.get_zlim()

        assert xlim == (0, 50)
        assert ylim == (0, 50)
        assert zlim == (0, 30)


class TestVisualizationHelper:
    def test_color_generation(self):
        """Test distinct color generation."""
        colors = VisualizationHelper.generate_distinct_colors(5)
        assert len(colors) == 5
        assert all(len(color) == 3 for color in colors)
        assert all(0 <= c <= 1 for color in colors for c in color)

    def test_energy_indicator(self):
        """Test energy indicator drawing."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        VisualizationHelper.draw_energy_indicator(
            ax,
            position=np.array([25, 25, 15]),
            energy=75.0
        )

        # Check indicator was drawn
        assert len(ax.lines) > 0

    def test_drone_trail(self):
        """Test drone trail drawing."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        positions = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        VisualizationHelper.draw_drone_trail(
            ax,
            positions=positions,
            color=(0.2, 0.5, 0.8)
        )

        # Check trail lines were drawn
        assert len(ax.lines) > 0