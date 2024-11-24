from elastic_mesh.core.elastic_mesh import ElasticMesh
from elastic_mesh.simulation.visualization import MeshVisualizer
from elastic_mesh.utils.weather import WeatherPattern
import numpy as np
from rich.console import Console

console = Console()


def run_weather_simulation():
    """Demonstrate weather pattern effects on drone behavior."""
    console.print("[bold blue]Starting Weather Pattern Simulation[/bold blue]")

    # Initialize mesh with custom weather patterns
    mesh = ElasticMesh(space_size=(50, 50, 30))

    # Create interesting weather patterns
    mesh.weather_system.create_thermal(
        center=np.array([25, 25, 0]),
        strength=1.2,
        radius=10.0
    )

    mesh.weather_system.create_wind_current(
        start=np.array([0, 25, 15]),
        direction=np.array([1, 0, 0]),
        strength=0.8,
        width=8.0
    )

    mesh.weather_system.create_turbulence(
        center=np.array([35, 35, 20]),
        intensity=0.9,
        size=5.0
    )

    # Add drones in interesting positions relative to weather
    scenarios = [
        ("drone_1", np.array([5, 25, 15]), np.array([45, 25, 15])),  # Through wind current
        ("drone_2", np.array([25, 25, 5]), np.array([25, 25, 25])),  # Through thermal
        ("drone_3", np.array([35, 35, 10]), np.array([35, 35, 25])),  # Through turbulence
        ("drone_4", np.array([10, 10, 20]), np.array([40, 40, 20])),  # Diagonal path
    ]

    for drone_id, start, goal in scenarios:
        mesh.add_drone(drone_id, start, goal)
        console.print(f"[green]Added {drone_id} to simulation[/green]")

    # Create visualizer with weather display enabled
    visualizer = MeshVisualizer(mesh)
    anim = visualizer.create_animation(
        duration=15,
        save_path='weather_simulation.gif'
    )

    # Generate analytics focusing on weather impact
    report = mesh.analytics.generate_report(save_path='weather_impact_report.csv')
    mesh.analytics.plot_metrics(save_path='weather_impact_metrics.png')

    console.print("\n[bold green]Weather Simulation Complete![/bold green]")
    console.print(f"Average Weather Impact: {report.attrs['summary']['avg_weather_impact']:.2f}")


if __name__ == "__main__":
    run_weather_simulation()