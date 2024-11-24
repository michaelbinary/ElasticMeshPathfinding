import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


class MeshReporter:
    """
    Generates comprehensive reports and visualizations for mesh simulation analytics.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_full_report(self, metrics_history: Dict, drone_states: Dict) -> str:
        """Generate a complete simulation report with analytics and visualizations."""
        report_dir = self.output_dir / f"report_{self.timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate all report components
        self._generate_metrics_report(metrics_history, report_dir)
        self._generate_drone_report(drone_states, report_dir)
        self._generate_visualizations(metrics_history, report_dir)
        report_path = self._compile_html_report(report_dir)

        return str(report_path)

    def _generate_metrics_report(self, metrics_history: Dict, report_dir: Path):
        """Generate detailed metrics analysis."""
        df = pd.DataFrame(metrics_history)

        # Calculate summary statistics
        summary = {
            'network_efficiency': {
                'mean': df['network_efficiency'].mean(),
                'max': df['network_efficiency'].max(),
                'min': df['network_efficiency'].min(),
                'std': df['network_efficiency'].std()
            },
            'mesh_density': {
                'mean': df['mesh_density'].mean(),
                'max': df['mesh_density'].max(),
                'min': df['mesh_density'].min(),
                'std': df['mesh_density'].std()
            },
            'system_energy': {
                'mean': df['system_energy'].mean(),
                'max': df['system_energy'].max(),
                'min': df['system_energy'].min(),
                'std': df['system_energy'].std()
            }
        }

        # Save summary and full metrics
        df.to_csv(report_dir / 'metrics.csv', index=False)
        with open(report_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

    def _generate_drone_report(self, drone_states: Dict, report_dir: Path):
        """Generate detailed per-drone analytics."""
        drone_stats = {}

        for drone_id, state in drone_states.items():
            drone_stats[drone_id] = {
                'total_distance': state.distance_traveled,
                'avg_connections': np.mean(state.connection_history),
                'final_energy': state.energy,
                'time_to_goal': state.time_to_goal,
                'connection_stability': np.std(state.connection_history)
            }

        # Save drone statistics
        with open(report_dir / 'drone_statistics.json', 'w') as f:
            json.dump(drone_stats, f, indent=4)

    def _generate_visualizations(self, metrics_history: Dict, report_dir: Path):
        """Generate visualization plots for key metrics."""
        plt.style.use('seaborn')

        # Create subplots
        fig = plt.figure(figsize=(20, 15))

        # Network Efficiency Plot
        plt.subplot(2, 2, 1)
        plt.plot(metrics_history['network_efficiency'], linewidth=2)
        plt.title('Network Efficiency Over Time', fontsize=12, pad=10)
        plt.xlabel('Time Step')
        plt.ylabel('Efficiency')
        plt.grid(True, alpha=0.3)

        # Mesh Density Plot
        plt.subplot(2, 2, 2)
        plt.plot(metrics_history['mesh_density'], linewidth=2)
        plt.title('Mesh Density Over Time', fontsize=12, pad=10)
        plt.xlabel('Time Step')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

        # System Energy Plot
        plt.subplot(2, 2, 3)
        plt.plot(metrics_history['system_energy'], linewidth=2)
        plt.title('System Energy Over Time', fontsize=12, pad=10)
        plt.xlabel('Time Step')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)

        # Weather Impact Plot
        plt.subplot(2, 2, 4)
        plt.plot(metrics_history['weather_impact'], linewidth=2)
        plt.title('Weather Impact Over Time', fontsize=12, pad=10)
        plt.xlabel('Time Step')
        plt.ylabel('Impact')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / 'metrics_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _compile_html_report(self, report_dir: Path) -> Path:
        """Compile all report components into an HTML report."""
        # Load data
        with open(report_dir / 'summary.json', 'r') as f:
            summary = json.load(f)
        with open(report_dir / 'drone_statistics.json', 'r') as f:
            drone_stats = json.load(f)

        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <title>Elastic Mesh Simulation Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; }}
                .metric-box {{ 
                    background: #f7f9fc;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px;
                    min-width: 200px;
                }}
                table {{ 
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{ 
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{ background-color: #f5f6fa; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Elastic Mesh Simulation Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>System Performance Summary</h2>
            <div class="metrics-container">
                {self._generate_metric_boxes(summary)}
            </div>

            <h2>Drone Performance Analysis</h2>
            {self._generate_drone_table(drone_stats)}

            <h2>Visualization</h2>
            <img src="metrics_visualization.png" alt="Metrics Visualization">
        </body>
        </html>
        """

        # Save HTML report
        report_path = report_dir / 'report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)

        return report_path

    def _generate_metric_boxes(self, summary: Dict) -> str:
        """Generate HTML for metric summary boxes."""
        boxes = []
        for metric, values in summary.items():
            boxes.append(f"""
                <div class="metric-box">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>Mean: {values['mean']:.3f}</p>
                    <p>Max: {values['max']:.3f}</p>
                    <p>Min: {values['min']:.3f}</p>
                    <p>Std: {values['std']:.3f}</p>
                </div>
            """)
        return '\n'.join(boxes)

    def _generate_drone_table(self, drone_stats: Dict) -> str:
        """Generate HTML table for drone statistics."""
        return f"""
        <table>
            <tr>
                <th>Drone ID</th>
                <th>Distance Traveled</th>
                <th>Avg Connections</th>
                <th>Final Energy</th>
                <th>Time to Goal</th>
                <th>Connection Stability</th>
            </tr>
            {''.join(
            f'''
                <tr>
                    <td>{drone_id}</td>
                    <td>{stats['total_distance']:.2f}</td>
                    <td>{stats['avg_connections']:.2f}</td>
                    <td>{stats['final_energy']:.2f}%</td>
                    <td>{stats['time_to_goal']:.2f}s</td>
                    <td>{stats['connection_stability']:.2f}</td>
                </tr>
                '''
            for drone_id, stats in drone_stats.items()
        )}
        </table>
        """