# next_best_view_rl

# Setup

1. Clone the repository: `git clone --recurse-submodules ...`
1. In 'third_party/zed-ros-wrapper': `git checkout devel`
1. Install mujoco
1. `sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libosmesa6-dev libgl1-mesa-glx libglfw3'
1. Create virtual environment: `conda create --name nbv_env python=3.7`
1. Activate virual environment: 'conda activate nbv_env'
1. Install requirements: `pip install -r requirements.txt`
1. Install setup.py: `pip install -e .`

# Execution

- modify configs in `config` with hydra framework
- start RL training: `python3 scripts/train.py`
- evaluate RL model: `python3 scripts/evaluate.py`
