# CALFQ-TD3: Constrained Actor Learning with Feasibility Quantification for TD3

![Demo](calf-td3-demo.gif)

## Overview

This repository implements CALFQ-TD3, a novel reinforcement learning algorithm that combines Constrained Actor Learning with Feasibility Quantification (CALFQ) and Twin Delayed Deep Deterministic Policy Gradient (TD3). The project focuses on training agents to control an underwater drone environment while respecting safety constraints.

## Features

- Implementation of CALFQ-TD3 algorithm
- Underwater drone simulation environment
- Training and evaluation scripts
- Video generation and analysis tools
- MLflow integration for experiment tracking

## Requirements

- Python >= 3.13
- PyTorch >= 2.6.0
- Stable-Baselines3 == 2.0.0
- Additional dependencies listed in `pyproject.toml`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/calfq-td3.git
cd calfq-td3
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Training

To train the agent using CALFQ-TD3:
```bash
./run/train_calfq_5seeds.sh
```

To train using standard TD3:
```bash
./run/train_td3_5seeds.sh
```

### Evaluation

To evaluate a trained model:
```bash
python run/eval_nominal.py
```

### Video Generation

To generate videos from training data:
```bash
python run/json_to_video.py
```

To stack multiple videos:
```bash
./run/stack_videos.sh
```

## Project Structure

```
calfq-td3/
├── src/
│   ├── envs/
│   │   └── underwaterdrone.py  # Underwater drone environment
│   ├── utils/                  # Utility functions
│   └── controller.py          # Controller implementation
├── run/
│   ├── train_calfq.py         # CALFQ-TD3 training script
│   ├── train_td3.py           # TD3 training script
│   ├── eval_nominal.py        # Evaluation script
│   └── json_to_video.py       # Video generation script
├── gfx/                       # Graphics and visualization
└── pyproject.toml            # Project dependencies
```

## Results

The trained agents can be evaluated using the provided evaluation scripts. Results are tracked using MLflow and can be visualized in the `analysis.ipynb` notebook.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
