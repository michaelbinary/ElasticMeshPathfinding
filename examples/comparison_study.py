from elastic_mesh.core.elastic_mesh import ElasticMesh
from elastic_mesh.simulation.visualization import MeshVisualizer
from elastic_mesh.simulation.animation import AnimationManager
import numpy as np
from rich.console import Console
import pandas as pd
import matplotlib.pyplot as plt

console = Console()


def run_comparison_study():
    """Compare different mesh configurations and parameters."""
    console.print("[bold blue]Starting Mesh Configuration Comparison Study[/bold blue]")

    # Define different configurations to compare
    configurations = {
        'baseline': {
            'k_spring': 0.5,
            'k_goal': 0.3,
            'damping': 0.3,
            'max_connection_distance': 8.0
        },
        'tight_formation': {
            'k_spring': 0.8,
            'k_goal': 0.2,
            'damping': 0.4,
            'max_connection_distance': 6.0
        },
        'loose_formation': {
            'k_spring': 0.3,
            'k_goal': 0.4,
            'damping': 0.2,
            'max_connection_distance': 12.0
        }
    }

    results = {}
    animations = []

    # Run simulations for each configuration
    for config_name, params in configurations.items():
        console.print(f"\n[yellow]Testing {config_name} configuration...[/yellow]")

        # Initialize mesh with configuration
        mesh = ElasticMesh(space_size=(50, 50, 30))
        mesh.physics_params.update(params)

        # Add identical drone formation to each configuration
        formation = [
            ("drone_1", np.array([10, 10, 15]), np.array([40, 40, 15])),
            ("drone_2", np.array([40, 10, 15]), np.array([10, 40, 15])),
            ("drone_3", np.array([25, 10, 15]), np.array([25, 40, 15])),
            ("drone_4", np.array([10, 25, 15]), np.array([40, 25, 15]))
        ]

        for drone_id, start, goal in formation:
            mesh.add_drone(drone_id, start, goal)

        # Create visualization and capture frames
        visualizer = MeshVisualizer(mesh)
        anim = visualizer.create_animation(
            duration=15,
            save_path=f'comparison_{config_name}.gif'
        )
        animations.append((config_name, visualizer.frames))

        # Collect metrics
        report = mesh.analytics.generate_report()
        results[config_name] = report.attrs['summary']

    # Create side-by-side comparison animation
    animation_manager = AnimationManager()
    animation_manager.create_side_by_side_comparison(
        animations[0][1],  # baseline frames
        animations[1][1],  # tight formation frames
        'formation_comparison.mp4',
        labels=('Baseline', 'Tight Formation')
    )

    # Generate comparison plots
    plt.figure(figsize=(15, 10))

    # Network Efficiency Comparison
    plt.subplot(2, 2, 1)
    efficiencies = [r['peak_network_efficiency'] for r in results.values()]
    plt.bar(configurations.keys(), efficiencies)
    plt.title('Peak Network Efficiency')
    plt.xticks(rotation=45)

    # Mesh Density Comparison
    plt.subplot(2, 2, 2)
    densities = [r['avg_mesh_density'] for r in results.values()]
    plt.bar(configurations.keys(), densities)
    plt.title('Average Mesh Density')
    plt.xticks(rotation=45)

    # Energy Efficiency Comparison
    plt.subplot(2, 2, 3)
    energies = [r['final_avg_energy'] for r in results.values()]
    plt.bar(configurations.keys(), energies)
    plt.title('Final Average Energy')
    plt.xticks(rotation=45)

    # Distance Covered Comparison
    plt.subplot(2, 2, 4)
    distances = [r['total_distance_covered'] for r in results.values()]
    plt.bar(configurations.keys(), distances)
    plt.title('Total Distance Covered')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('configuration_comparison.png')

    # Print summary
    console.print("\n[bold green]Comparison Study Complete![/bold green]")
    console.print("\n[bold]Configuration Rankings:[/bold]")

    # Rank configurations by different metrics
    metrics = ['network_efficiency', 'mesh_density', 'energy_efficiency']
    for metric in metrics:
        sorted_configs = sorted(results.items(),
                                key=lambda x: x[1].get(f'avg_{metric}', 0),
                                reverse=True)
        console.print(f"\n{metric.replace('_', ' ').title()}:")
        for i, (config, _) in enumerate(sorted_configs, 1):
            console.print(f"{i}. {config}")


if __name__ == "__main__":
    run_comparison_study()