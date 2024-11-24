# Elastic Mesh Simulation API Documentation

## Core Components

### ElasticMesh

The main simulation class that handles drone coordination and physics.

```python
mesh = ElasticMesh(space_size=(50, 50, 30))
```

#### Parameters
- `space_size`: Tuple[float, float, float] - (x, y, z) dimensions of simulation space

#### Key Methods

##### add_drone(drone_id: str, start: np.ndarray, goal: np.ndarray)
Add a new drone to the mesh system.
```python
mesh.add_drone("drone_1", np.array([0, 0, 0]), np.array([10, 10, 10]))
```

##### update(dt: float)
Update the simulation state for one time step.
```python
mesh.update(0.1)  # Update with 0.1 second time step
```

##### get_system_state() -> dict
Get current state of the entire system.
```python
state = mesh.get_system_state()
```

### DroneState

Data class representing individual drone state.

```python
@dataclass
class DroneState:
    position: np.ndarray        # Current 3D position
    velocity: np.ndarray        # Current 3D velocity
    goal: np.ndarray           # Target position
    connections: List[str]      # Connected drone IDs
    energy: float              # Current energy level (0-100)
    weather_influence: float    # Weather effect multiplier
```

## Visualization Components

### MeshVisualizer

Handles 3D visualization of the simulation.

```python
visualizer = MeshVisualizer(mesh, figure_size=(15, 10))
```

#### Key Methods

##### create_animation(duration: float, save_path: str)
Create and save an animation of the simulation.

```python
visualizer.create_animation(
    duration=10,
    save_path='../simulation.gif'
)
```

##### save_frame(path: str)
Save current visualization frame.
```python
visualizer.save_frame('frame.png')
```

### AnimationManager

Manages advanced animation features.

```python
manager = AnimationManager(fps=30, quality=95)
```

#### Key Methods

##### save_gif(filename: str)
Save captured frames as optimized GIF.
```python
manager.save_gif('animation.gif')
```

##### create_side_by_side_comparison(frames1, frames2, filename: str)
Create comparison video of two simulations.
```python
manager.create_side_by_side_comparison(
    frames1, frames2,
    'comparison.mp4',
    labels=('Sim 1', 'Sim 2')
)
```

## Analytics Components

### MeshAnalytics

Handles data collection and analysis.

```python
analytics = mesh.analytics
```

#### Key Methods

##### update_metrics(drones: Dict[str, DroneState], system_energy: float)
Update analytics for current time step.
```python
analytics.update_metrics(mesh.drones, system_energy)
```

##### generate_report(save_path: str = None) -> pd.DataFrame
Generate comprehensive analytics report.
```python
report = analytics.generate_report('report.csv')
```

## Weather System

### WeatherSystem

Manages weather patterns and environmental effects.

```python
weather = mesh.weather_system
```

#### Key Methods

##### create_thermal(center: np.ndarray, strength: float, radius: float)
Create thermal updraft.
```python
weather.create_thermal(
    center=np.array([25, 25, 0]),
    strength=1.2,
    radius=10.0
)
```

##### create_wind_current(start: np.ndarray, direction: np.ndarray, strength: float)
Create directional wind current.
```python
weather.create_wind_current(
    start=np.array([0, 25, 15]),
    direction=np.array([1, 0, 0]),
    strength=0.8
)
```

## Utility Components

### VisualizationHelper

Provides enhanced visualization features.

```python
viz_helper = VisualizationHelper()
```

#### Key Methods

##### draw_drone_trail(ax: Axes3D, positions: np.ndarray, color: Tuple[float, float, float])
Draw motion trail behind drone.
```python
viz_helper.draw_drone_trail(ax, drone_positions, color=(0.2, 0.5, 0.8))
```

##### draw_energy_indicator(ax: Axes3D, position: np.ndarray, energy: float)
Visualize drone energy level.
```python
viz_helper.draw_energy_indicator(ax, drone.position, drone.energy)
```