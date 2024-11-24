elastic-mesh-simulation/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── images/
│   ├── api.md
│   └── analytics.md
├── examples/
│   ├── basic_simulation.py
│   └── custom_weather_patterns.py
├── tests/
│   ├── __init__.py
│   ├── test_elastic_mesh.py
│   ├── test_simulation.py
│   └── test_analytics.py
└── elastic_mesh/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── drone_state.py
    │   └── elastic_mesh.py
    ├── simulation/
    │   ├── __init__.py
    │   ├── visualization.py
    │   └── animation.py
    ├── analytics/
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── reporting.py
    └── utils/
        ├── __init__.py
        ├── weather.py
        └── visualization_helpers.py