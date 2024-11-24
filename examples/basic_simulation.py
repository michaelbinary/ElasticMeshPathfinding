from core.elastic_mesh import ElasticMesh
from simulation.visualization import MeshVisualizer
import numpy as np
from rich.console import Console

console = Console()


def run_basic_simulation():
    """Run a basic simulation with default parameters."""
    console.print("[bold blue]Starting Basic Elastic Mesh Simulation[/bold blue]")

    # Initialize the mesh
    mesh = ElasticMesh(space_size=(50, 50, 30))

    # Add some drones
    scenarios = [
        ("drone_1", np.array([10, 10, 5]), np.array([40, 40, 25])),
        ("drone_2", np.array([40, 10, 15]), np.array([10, 40, 5])),
        ("drone_3", np.array([25, 5, 20]), np.array([25, 45, 10])),
    ]

    for drone_id, start, goal in scenarios:
        mesh.add_drone(drone_id, start, goal)
        console.print(f"[green]Added {drone_id} to simulation[/green]")

    # Create visualizer and run simulation
    visualizer = MeshVisualizer(mesh)
    anim = visualizer.create_animation(
        duration=10,
        save_path='basic_simulation.gif'
    )

    # Show the animation
    visualizer.show()

    # Generate and save analytics
    report = mesh.analytics.generate_report(save_path='basic_simulation_report.csv')

    # Print summary with error handling
    console.print("\n[bold green]Simulation Complete![/bold green]")
    console.print("\n[bold]Performance Summary:[/bold]")

    try:
        summary = report.attrs['summary']
        console.print(f"Network Efficiency: {summary.get('network_efficiency', {}).get('mean', 0):.2f}")
        console.print(f"Mesh Density: {summary.get('mesh_density', {}).get('mean', 0):.2f}")
        console.print(f"System Energy: {summary.get('system_energy', {}).get('mean', 0):.2f}")

        # Additional metrics if available
        if 'total_distance_covered' in summary:
            console.print(f"Total Distance Covered: {summary['total_distance_covered']:.2f}m")
        if 'final_avg_energy' in summary:
            console.print(f"Final Average Energy: {summary['final_avg_energy']:.2f}%")

    except (KeyError, AttributeError) as e:
        console.print("[yellow]Note: Some analytics data not available[/yellow]")
        console.print(f"Available metrics in report: {list(report.attrs['summary'].keys())}")


if __name__ == "__main__":
    run_basic_simulation()