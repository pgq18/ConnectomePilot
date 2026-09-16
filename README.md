# ConnectomePilot

**English** | [简体中文](README.zh-CN.md)

**Explore robot residual control and reinforcement learning with a real fly connectome.**

ConnectomePilot connects the MaleCNS anatomical connectivity graph to a simulated mobile robot. Sensor inputs drive neural activity, and a policy readout produces corrections to a base navigation controller. The project includes an interactive Chinese-language demo, whole-brain and subgraph models, PPO baselines, learnable connection weights, staged training, and video export.

The best-performing training approach so far is to **jointly train internal connections and readout for 65,536 steps, then freeze the connections and continue training the readout to 4,194,304 total steps.**

## Current results

Evaluated on the same 500 held-out test maps with three training seeds:

| Measurement | Success rate |
|---|---:|
| Mean across three seeds at 1,048,576 steps | 85.1% |
| Mean across three seeds at 4,194,304 steps | 88.6% |
| Mean test result after selecting a checkpoint per seed using validation maps | 89.6% |
| Single model selected across seeds using the predefined validation rule | 89.4% |
| Highest test point observed retrospectively, seed 72 | 92.6% (463/500) |

The 92.6% result describes the highest observed test point; the single model selected by the predefined rule achieved 89.4%. Longer training did not improve performance monotonically, and average changes in angular velocity commands increased. See the [experiment report](docs/STAGED_4M_TRIAL.md) and [predefined protocol](docs/STAGED_4M_PROTOCOL.md).

The model with the highest observed test score reached the goal on two new demonstration maps: map 5900000 in 9.4 seconds and map 5900001 in 11.2 seconds of simulated time. Videos are available in the [v0.1.0 release](https://github.com/pgq18/ConnectomePilot/releases/tag/v0.1.0).

## Four ways to use the project

| Goal | Entry point | What it provides |
|---|---|---|
| Try interactive obstacle avoidance | `./launch.sh --open` | Base controller, rule-based baseline, whole-brain LIF model, and early SB3 PPO policies |
| Learn internal connection strengths | `scripts/train_plasticity.py` | Readout-only PPO, joint connection/readout PPO, and local three-factor learning |
| Reproduce staged training | [Training guide](docs/TRAINING.md) | 65k joint → 262k frozen → 1M → 4M; the final stage supports three parallel training processes |
| Run a policy and export video | [Video guide](docs/VIDEOS.md) | Record an actual policy rollout, render frames, then encode and validate the video |

The web interface uses the earlier controllers. The latest 4M subgraph policy runs through offline evaluation and video tools and is not yet available in the web menu.

## Quick start

Install Conda first. All Python packages belong in the project-specific environment. Run these commands from the repository root:

```bash
git clone https://github.com/pgq18/ConnectomePilot.git
cd ConnectomePilot
PROJECT_DIR="$(pwd)"
conda env create --prefix "$PROJECT_DIR/.conda" --file environment.yml
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements-data.txt
./launch.sh --open
```

Open <http://127.0.0.1:8765>. The base controller and rule-based obstacle avoidance work without downloading connectome data. Fly-brain mode remains disabled until the data is available. On macOS, you can also double-click `launch.command`.

### Set goals interactively

Click an empty spot on the map to place a goal. The displayed coordinates use meters. You can change the goal while the robot is moving; when paused, select a goal and press **Continue / Start**.

Each goal change starts a new leg from the robot's current position and resets its trajectory, timer, and metrics. Goals must leave enough room for the robot around obstacles and walls. After a collision, reset the scene first. Resetting the same map or switching controllers preserves your chosen goal; changing the map restores the default goal. Use **Restore default goal** to switch back explicitly.

The published benchmark scores use the original fixed-goal tasks. Performance on arbitrary interactive goals has not been benchmarked.

### Prepare the measured connectome

```bash
"$PROJECT_DIR/.conda/bin/python" scripts/prepare_data.py --check-network
"$PROJECT_DIR/.conda/bin/python" scripts/prepare_data.py
```

The script downloads approximately 1.2 GB of raw tables from the official MaleCNS storage and produces sparse weights, annotations, and provenance hashes. Preprocessing requires additional RAM and disk space. The full graph loaded by this project contains **166,700 neurons and 25,582,938 directed connections**.

Downloads support validation, retries, and resumption. If a proxy is needed, copy `config/network.example.json` to `config/network.local.json` and edit it, or pass `--proxy http://127.0.0.1:PORT`. Use `--no-proxy` for a direct connection. On macOS, `prepare-and-launch.command` prepares the data before opening the web interface.

### Install training dependencies

```bash
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements-rl.txt
```

The web demo and small sensor-based PPO runs can use a CPU. The current staged trainers require NVIDIA CUDA. See the [training guide](docs/TRAINING.md) and [workstation guide](docs/WORKSTATION.md) for GPU setup, measured resource use, and reproduction commands. The `requirements-rl*.lock.txt` files record the original experiment environments; macOS and Linux snapshots are not interchangeable.

## How it works

```mermaid
flowchart LR
    R[Range and closing speed] --> E[Fixed directional encoding]
    E --> B[Measured MaleCNS subgraph]
    B --> D[Activity of 1314 descending neurons]
    D --> P[PPO readout]
    C[Goal and robot state] --> P
    G[Goal direction] --> A[Base navigation action]
    P --> X[Speed and turning residuals]
    A --> S[Combine and bound actions]
    X --> S
    S --> W[Simulated 2D robot]
    W --> R
```

The latest policy uses **4,096 neurons and 459,342 permitted connections**. It retains measured connection locations and learns their strengths during joint training. After freezing, neural propagation still runs at every control step; PPO updates only the **3,960 actor/critic readout parameters**.

Connection locations come from anatomical data. Transmitter signs, normalized weights, sensory encoding, simplified neural dynamics, and action mappings involve engineering assumptions. The model does not contain a biological fly's learned memories or reproduce a complete biological brain. Validation currently covers a 2D simulation; robot dynamics simulators and hardware integration remain future work.

## Code and documentation

```text
flylab/        Simulation, controllers, neural models, learning, and state recovery
scripts/       Data preparation, training, evaluation, audits, plots, and video tools
tests/         Environment, propagation, gradients, freezing, resumption, and selection
web/           Chinese-language interactive interface
config/        Public network configuration example
docs/          Architecture, usage guides, experiment protocols, and reports
data/          Local downloaded and processed data (excluded from Git)
results/       Local models, logs, plots, and videos (excluded from Git)
```

Detailed guides and experiment reports are currently in Chinese:

- [Architecture and module responsibilities](docs/ARCHITECTURE.md)
- [Training and evaluation reproduction](docs/TRAINING.md)
- [Video export](docs/VIDEOS.md)
- [Documentation and experiment index](docs/README.md)
- [Robot interfaces and extensions](docs/ROBOTICS.md)

See also the English [contribution guide](CONTRIBUTING.md).

Raw data and model checkpoints are not distributed with the Git source. Prepare the data from the official source, then follow the training guide. Historical reports preserve measured results; the referenced `results/` artifacts reside in the experiment workspace.

## Validation

After installing the data and training dependencies:

```bash
"$PROJECT_DIR/.conda/bin/python" -m unittest discover -s tests -v
```

Tests use explicitly labeled small synthetic graphs to verify interfaces, numerical calculations, and state recovery. These fixtures are not biological experiment results. GitHub Actions runs CPU tests in a dedicated Conda environment without downloading MaleCNS or training models.

## Sources and licensing

Code is released under the [MIT License](LICENSE). MaleCNS data has a separate [CC BY 4.0 license](https://male-cns.janelia.org/download/).

Data preparation and early simplified dynamics draw on [fly.ai](https://github.com/alextitonis/fly.ai). PPO with fixed connection locations and reward-modulated learning rules are inspired by [FlyDoom](https://github.com/eganeganegan/flydoom). See [third-party notices](docs/THIRD_PARTY.md) for other references and adaptation details.
