# 2D Swarm-vs-Swarm Simulation

A Python-based simulation environment for studying swarm dynamics and interactions between multiple swarms of agents in 2D space.

## Features

- Efficient spatial hashing for collision detection
- Configurable agent behaviors and team assignments
- Real-time visualization using matplotlib
- Extensible controller system for different movement strategies
- Performance optimized for large swarms (10,000+ agents)

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd swarm_sim
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation with default parameters:
```bash
python main.py
```

Configure simulation parameters through environment variables or command-line arguments:
```bash
python main.py --num-agents 1000 --world-size 1000
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Type checking: `mypy .`
- Lint code: `pylint swarm_sim/`

## Project Structure

```
swarm_sim/
├── environment/     # Core simulation environment
├── controllers/     # Agent movement controllers
├── tests/          # Test suite
├── main.py         # Main simulation runner
└── config.py       # Configuration management
```

## License

[Your chosen license] 