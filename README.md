# 🐦 Flappy Bird with NEAT AI

A Flappy Bird clone built with Pygame that features both a **human-playable mode** and an **AI training mode** powered by the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Features

- **Human Mode** — Classic Flappy Bird gameplay with keyboard controls
- **AI Training** — Watch NEAT evolve neural networks that learn to play the game
- **Checkpoint System** — Training automatically saves progress every 10 generations and resumes from the latest checkpoint
- **Best Genome Replay** — Save and replay the best-performing neural network

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/flappy-bird-neat.git
cd flappy-bird-neat

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Play the game yourself:**

```bash
python flappy_game.py
```

- Press **SPACE** to start and to jump
- Press **R** to restart after game over

**Train the AI:**

```bash
python ai_trainer.py
```

Choose `t` to start/resume training, or `b` to watch the best saved genome play.

## Project Structure

| File | Description |
|------|-------------|
| `flappy_game.py` | Game engine — `Bird`, `Pipe`, and `FlappyGame` classes, plus human-playable `main()` |
| `ai_trainer.py` | NEAT training loop, checkpoint management, and best-genome replay |
| `neat_config.txt` | NEAT algorithm configuration (population size, mutation rates, network topology) |

## How the AI Works

The NEAT algorithm evolves neural networks through natural selection:

1. **Population** — Each generation starts with 50 birds, each controlled by a unique neural network
2. **Inputs** — Each network receives 5 inputs: horizontal distance to next pipe, vertical distance to top/bottom of gap, bird velocity, and bird Y position
3. **Output** — A single output determines whether the bird jumps (> 0.5 = jump)
4. **Fitness** — Birds are rewarded for surviving (+0.1/frame) and passing pipes (+10), penalized for dying (-5)
5. **Evolution** — The best-performing networks are selected, mutated, and crossed over to produce the next generation

## Configuration

Key NEAT parameters in `neat_config.txt`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pop_size` | 50 | Birds per generation |
| `fitness_threshold` | 100000 | Target fitness to stop training |
| `num_inputs` | 5 | Neural network input neurons |
| `num_outputs` | 1 | Neural network output neurons |
| `max_stagnation` | 20 | Generations without improvement before a species is removed |

## License

This project is open source. Feel free to use and modify.
