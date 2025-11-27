# AlphaZero-OmniFive

AlphaZero-OmniFive applies the AlphaZero algorithm to Gomoku (Five in a Row), training a policy-value network purely through self-play data combined with Monte Carlo Tree Search (MCTS) for decision-making. Since Gomoku's state space is much smaller than Go or Chess, a competitive AI can be trained in just a few hours on a PC with a CUDA-enabled GPU.

#### Differences Between AlphaGo and AlphaGo Zero

- **AlphaGo**: Combines expert game records, hand-crafted features, and move prediction with MCTS, further enhanced through self-play.
- **AlphaGo Zero**: Starts from scratch using only game rules for self-play, employs residual convolutional networks to output both policy and value simultaneously with MCTS; abandons hand-crafted features and human game records for a simpler architecture with more efficient training and inference, surpassing AlphaGo in strength.

![playout400](playout400.gif)

#### System Requirements

- Python >= 3.9
- PyTorch >= 2.0 (CUDA-capable GPU driver environment required)
- numpy >= 1.24

#### Network Architecture

The project uses a **ResNet (Residual Neural Network)** architecture optimized for Gomoku:

- **Initial Conv Block**: Converts 4-channel board state to 128-channel features
- **Residual Tower**: Configurable number of residual blocks (default: 6) for deep feature extraction
- **Policy Head**: Outputs move probabilities via 1×1 convolution and fully-connected layers
- **Value Head**: Estimates board game value (-1 to 1) via separate tower of layers

ResNet's skip connections enable effective deep networks without training degradation, leading to better feature extraction and improved game understanding compared to plain CNNs.

#### Initial Setup

```bash
git clone https://github.com/Suyw-0123/AlphaZero-OmniFive.git
cd AlphaZero-OmniFive
```

## Play Against the Model

```bash
python human_play.py
```

## Train the Model

```bash
python train.py
```

Training workflow includes:

1. Self-play to collect game records with rotation and flipping augmentation.
2. Mini-batch updates to the policy-value network.
3. Periodic evaluation against a pure MCTS opponent; if win rate improves, overwrite `best_policy.model`.

Output models:

- `current_policy.model`: The most recently trained network.
- `best_policy.model`: The network with the best evaluation performance so far.

## config.json Training Parameters

### Board Configuration

| Parameter | Description |
| --- | --- |
| `board_width` / `board_height` | Board dimensions; adjust `n_in_row` accordingly when changing. |
| `n_in_row` | Win condition (five in a row). Determines game difficulty along with board size. |

### Network Configuration

| Parameter | Description |
| --- | --- |
| `num_channels` | Number of feature channels in residual blocks (default: 128). Higher values increase model capacity. |
| `num_res_blocks` | Number of residual blocks in the tower (default: 6). More blocks enable deeper feature extraction. |

### Training Configuration

| Parameter | Description |
| --- | --- |
| `learn_rate` | Initial Adam learning rate. Dynamically scaled by `lr_multiplier` based on KL divergence. |
| `lr_multiplier` | Multiplicatively adjusted when KL exceeds or falls below thresholds, controlling learning rate decay or recovery. |
| `temp` | Temperature during self-play, controlling move exploration; can be lowered later to reduce randomness. |
| `n_playout` | Number of MCTS simulations per move. Higher values increase strength but also inference time. |
| `c_puct` | MCTS exploration coefficient, balancing high visit counts and high-scoring nodes. |
| `buffer_size` | Self-play data buffer capacity; larger values retain more historical games for training. |
| `batch_size` | Number of samples per gradient update. Adjust based on GPU memory; recommend 512-640 for 8GB GPUs. |
| `play_batch_size` | Number of games generated per self-play round. |
| `epochs` | Number of mini-batch iterations per update, improving convergence speed. |
| `kl_targ` | Target KL divergence, limiting policy change between old and new, working with `lr_multiplier` to control step size. |
| `check_freq` | Frequency (in batches) for MCTS evaluation and model saving. |
| `game_batch_num` | Training loop upper limit; Ctrl+C saves the current best model. |
| `pure_mcts_playout_num` | Number of simulations for the pure MCTS opponent during evaluation; higher values make evaluation stricter. |

> **GPU Memory Optimization**: ResNet with 128 channels and 6 blocks typically requires ~2-3GB VRAM for batch_size=256. If GPU memory is insufficient, reduce `batch_size` to 512 or 384, and lower `num_channels` to 64 or `num_res_blocks` to 4 to reduce model size.

## References

- Special thanks to [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git) for providing the core codebase.

- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016)
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017)
