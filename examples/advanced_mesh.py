from core.elastic_mesh import ElasticMesh
from simulation.visualization import MeshVisualizer
from utils.visualization_helpers import VisualizationHelper
import numpy as np
from rich.console import Console
from datetime import datetime

console = Console()


def run_advanced_simulation():
    """Demonstrate advanced features of the elastic mesh simulation."""
    console.print("[bold blue]Starting Advanced Elastic Mesh Simulation[/bold blue]")

    # Initialize mesh with custom physics parameters
    mesh = ElasticMesh(space_size=(50, 50, 30))
    mesh.physics_params.update({
        'k_spring': 0.7,  # Stronger spring connections
        'k_goal': 0.4,  # Stronger goal attraction
        'damping': 0.25,  # Less damping for more dynamic movement
        'max_connection_distance': 12.0  # Longer connection range
    })

    # Create a complex initial formation
    formation = [
        ("leader", np.array([25, 25, 15]), np.array([25, 25, 25])),
        ("scout_1", np.array([20, 20, 15]), np.array([30, 30, 25])),
        ("scout_2", np.array([30, 20, 15]), np.array([20, 30, 25])),
        ("relay_1", np.array([15, 25, 15]), np.array([35, 25, 25])),
        ("relay_2", np.array([35, 25, 15]), np.array([15, 25, 25])),
        ("support_1", np.array([25, 15, 15]), np.array([25, 35, 25])),
        ("support_2", np.array([25, 35, 15]), np.array([25, 15, 25])),
        ("observer", np.array([25, 25, 25]), np.array([25, 25, 15]))
    ]

    for drone_id, start, goal in formation:
        mesh.add_drone(drone_id, start, goal)
        console.print(f"[green]Added {drone_id} to formation[/green]")

    # Create visualizer with enhanced settings
    visualizer = MeshVisualizer(mesh)
    visualizer.animation_settings.update({
        'fps': 30,
        'quality': 95,
        'dpi': 150
    })

    # Create animation with telemetry overlay
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    anim = visualizer.create_animation(
        duration=20,
        save_path=f'advanced_simulation_{timestamp}.gif'
    )

    # Generate detailed analytics
    report = mesh.analytics.generate_report(save_path=f'advanced_report_{timestamp}.csv')
    mesh.analytics.plot_metrics(save_path=f'advanced_metrics_{timestamp}.png')

    # Print detailed summary
    console.print("\n[bold green]Advanced Simulation Complete![/bold green]")
    console.print("\n[bold]Performance Metrics:[/bold]")
    console.print(f"Peak Network Efficiency: {report.attrs['summary']['peak_network_efficiency']:.2f}")
    console.print(f"Average Mesh Density: {report.attrs['summary']['avg_mesh_density']:.2f}")
    console.print(f"Total Distance Covered: {report.attrs['summary']['total_distance_covered']:.2f}m")
    console.print(f"Final Average Energy: {report.attrs['summary']['final_avg_energy']:.2f}%")


if __name__ == "__main__":
    run_advanced_simulation()