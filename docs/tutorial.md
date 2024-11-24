# Elastic Mesh Simulation Tutorial

## Getting Started

### Installation

```bash
# Clone the repository
cd elastic-mesh-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Basic Usage

Here's a simple example to get started:

```python
from elastic_mesh.core.elastic_mesh import ElasticMesh
from elastic_mesh.simulation.visualization import MeshVisualizer
import numpy as np

# Create mesh
mesh = ElasticMesh(space_size=(50, 50, 30))

# Add drones
mesh.add_drone(
    "drone_1",
    start=np.array([10, 10, 5]),
    goal=np.array([40, 40, 25])
)

# Visualize
visualizer = MeshVisualizer(mesh)
visualizer.create_animation(duration=10, save_path='sim.gif')
```

## Advanced Features

### Weather Patterns

Create complex weather conditions:

```python
# Create thermal updraft
mesh.weather_system.create_thermal(
    center=np.array([25, 25, 0]),
    strength=1.2,
    radius=10.0
)

# Add wind current
mesh.weather_system.create_wind_current(
    start=np.array([0, 25, 15]),
    direction=np.array([1, 0, 0]),
    strength=0.8
)
```

### Analytics

Track and analyze simulation performance:

```python
# Generate report
report = mesh.analytics.generate_report('analysis.csv')

# Plot metrics
mesh.analytics.plot_metrics('metrics.png')
```

### Custom Formations

Create complex drone formations:

```python
formation = [
    ("leader", np.array([25, 25, 15]), np.array([25, 25, 25])),
    ("scout_1", np.array([20, 20, 15]), np.array([30, 30, 25])),
    ("scout_2", np.array([30, 20, 15]), np.array([20, 30, 25]))
]

for drone_id, start, goal in formation:
    mesh.add_drone(drone_id, start, goal)
```

## Best Practices

1. **Space Size**: Choose appropriate space dimensions based on:
   - Number of drones
   - Desired formation complexity
   - Weather pattern sizes

2. **Physics Parameters**: Adjust for desired behavior:
   ```python
   mesh.physics_params.update({
       'k_spring': 0.7,        # Stronger connections
       'k_goal': 0.4,          # Stronger goal attraction
       'damping': 0.25,        # More dynamic movement
   })
   ```

3. **Connection Range**: Set based on formation needs:
   ```python
   mesh.physics_params['max_connection_distance'] = 12.0
   ```

4. **Weather Effects**: Use weather patterns strategically:
   - Thermals for vertical movement
   - Wind currents for horizontal assistance
   - Turbulence for testing resilience

## Troubleshooting

Common issues and solutions:

1. **Unstable Formations**
   - Decrease spring constant
   - Increase damping
   - Reduce connection distance

2. **Poor Performance**
   - Reduce number of drones
   - Simplify weather patterns
   - Increase time step

3. **Visualization Issues**
   - Adjust figure size
   - Reduce animation duration
   - Lower frame rate

## Advanced Topics

### Custom Analytics

Implement custom metrics:

```python
class CustomAnalytics(MeshAnalytics):
    def __init__(self):
        super().__init__()
        self.custom_metric = []
    
    def update_metrics(self, drones, system_energy):
        super().update_metrics(drones, system_energy)
        # Add custom metric
        self.custom_metric.append(
            self.calculate_custom_metric(drones)
        )
```

### Weather Pattern Creation

Create custom weather patterns:

```python
class CustomWeatherPattern(WeatherPattern):
    def __init__(self, center, intensity):
        super().__init__(
            center=center,
            intensity=intensity,
            pattern_type='custom'
        )
    
    def calculate_influence(self, position):
        # Custom influence calculation
        return custom_influence_function(position)
```