# AlphaZero-OmniFive

AlphaZero-OmniFive applies the AlphaZero algorithm to Gomoku (Five in a Row), training a policy-value network purely through self-play data combined with Monte Carlo Tree Search (MCTS) for decision-making. Since Gomoku's state space is much smaller than Go or Chess, a competitive AI can be trained in just a few hours on a PC with a CUDA-enabled GPU.

This repo is based on [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git), and makes the following modifications:

**Original Enhancements:**
- Changed the network architecture from CNN to ResNet
- Optimized MCTS and self-play modules by leveraging PyTorch CUDA acceleration
- Tuned training parameters specifically for large boards of size 9x9 and above
- Added the models trained using this parameter
- Added a new config.json file for centralized parameter management

**Latest Optimizations (New):**
- 🔥 **Squeeze-and-Excitation (SE) Blocks**: Channel attention mechanism for improved feature learning
- 🎯 **First Play Urgency (FPU)**: Enhanced MCTS exploration strategy for unvisited nodes
- 🛡️ **Gradient Clipping**: Training stability improvement preventing gradient explosion
- 📈 **Learning Rate Warmup**: Smoother training initialization phase
- 🔧 **Dropout Regularization**: Configurable dropout for preventing overfitting
- ⚡ **Virtual Loss Infrastructure**: Preparation for parallel MCTS simulations
- 📋 **Configuration Presets**: Ready-to-use optimized configs for different scenarios

#### Differences Between AlphaGo and AlphaGo Zero

- **AlphaGo**: Combines expert game records, hand-crafted features, and move prediction with MCTS, further enhanced through self-play.
- **AlphaGo Zero**: Starts from scratch using only game rules for self-play, employs residual convolutional networks to output both policy and value simultaneously with MCTS; abandons hand-crafted features and human game records for a simpler architecture with more efficient training and inference, surpassing AlphaGo in strength.

![playout400](playout400.gif)

#### System Requirements

- Python >= 3.9
- PyTorch >= 2.0 (CUDA-capable GPU driver environment required)
- numpy >= 1.24

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
# Using default config
python train.py

# Or use one of the optimized presets
python train.py --config configs/small_board_optimized.json
python train.py --config configs/large_board_optimized.json
python train.py --config configs/fast_training.json
python train.py --config configs/maximum_strength.json
```

**See [configs/README.md](configs/README.md) for detailed preset descriptions and customization tips.**

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
| `num_channels` | Number of feature channels in residual blocks. Higher values increase model capacity. |
| `num_res_blocks` | Number of residual blocks in the tower. More blocks enable deeper feature extraction. |
| `use_se` | Enable Squeeze-and-Excitation blocks for channel attention (default: true). Improves feature representation. |
| `dropout_rate` | Dropout rate for regularization (default: 0.1). Range: [0, 1). Prevents overfitting. |
| `l2_const` | L2 regularization coefficient (default: 1e-4). Weight decay for model parameters. |

### Training Configuration

| Parameter | Description |
| --- | --- |
| `learn_rate` | Initial Adam learning rate. Dynamically scaled by `lr_multiplier` based on KL divergence. |
| `lr_multiplier` | Multiplicatively adjusted when KL exceeds or falls below thresholds, controlling learning rate decay or recovery. |
| `temp` | Temperature during self-play, controlling move exploration; can be lowered later to reduce randomness. |
| `n_playout` | Number of MCTS simulations per move. Higher values increase strength but also inference time. |
| `c_puct` | MCTS exploration coefficient, balancing high visit counts and high-scoring nodes. |
| `fpu_reduction` | First Play Urgency reduction factor (default: 0.25). Balances exploration of unvisited nodes. |
| `buffer_size` | Self-play data buffer capacity; larger values retain more historical games for training. |
| `batch_size` | Number of samples per gradient update. Adjust based on GPU memory. |
| `play_batch_size` | Number of games generated per self-play round. |
| `epochs` | Number of mini-batch iterations per update, improving convergence speed. |
| `kl_targ` | Target KL divergence, limiting policy change between old and new, working with `lr_multiplier` to control step size. |
| `check_freq` | Frequency (in batches) for MCTS evaluation and model saving. |
| `game_batch_num` | Training loop upper limit; Ctrl+C saves the current best model. |
| `pure_mcts_playout_num` | Number of simulations for the pure MCTS opponent during evaluation. Higher values make evaluation stricter. |
| `use_gpu` | Whether to use GPU acceleration for training and inference. |
| `init_model` | Path to the initial model file to resume training from a checkpoint. |
| `grad_clip` | Gradient clipping threshold (default: 5.0). Prevents gradient explosion. Set to 0 to disable. |
| `lr_warmup_steps` | Number of steps for learning rate warmup (default: 1000). Stabilizes early training. Set to 0 to disable. |
| `use_cosine_annealing` | Enable cosine annealing learning rate schedule (default: false). For future implementation. |


### Human Play Configuration

| Parameter | Description |
| --- | --- |
| `model_file` | Path to the model file used for human vs AI games. |
| `start_player` | Set to 0 for human first, 1 for AI first. |
| `n_playout` | Number of MCTS simulations per move for the AI during human play. |
| `c_puct` | MCTS exploration coefficient for human play. |
| `fpu_reduction` | First Play Urgency reduction factor for human play (default: 0.25). |
| `use_gpu` | Whether to use GPU acceleration for inference during human play. |

## Optimization Guide

### New Optimizations in This Version

This version includes several advanced optimizations to improve the Gomoku AI:

#### 1. **Network Architecture Enhancements**
- **Squeeze-and-Excitation (SE) Blocks**: Improves channel-wise feature representation through attention mechanisms. Enable with `use_se=true`.
- **Dropout Regularization**: Prevents overfitting during training. Configure with `dropout_rate` (recommended: 0.1-0.3).
- **Configurable L2 Regularization**: Fine-tune weight decay with `l2_const`.

#### 2. **MCTS Improvements**
- **First Play Urgency (FPU)**: Better exploration of unvisited nodes. Controlled by `fpu_reduction` parameter.
- **Virtual Loss**: Prepared infrastructure for parallel MCTS simulations (reduces redundant exploration).

#### 3. **Training Stability**
- **Gradient Clipping**: Prevents gradient explosion with `grad_clip` parameter (recommended: 5.0).
- **Learning Rate Warmup**: Stabilizes early training with `lr_warmup_steps` (recommended: 1000 steps).

### Recommended Configurations

#### For Smaller Boards (8x8 or less)
```json
{
  "network": {
    "num_channels": 128,
    "num_res_blocks": 4,
    "use_se": true,
    "dropout_rate": 0.1
  },
  "training": {
    "n_playout": 400,
    "fpu_reduction": 0.25,
    "grad_clip": 5.0,
    "lr_warmup_steps": 500
  }
}
```

#### For Larger Boards (11x11 or more)
```json
{
  "network": {
    "num_channels": 256,
    "num_res_blocks": 6,
    "use_se": true,
    "dropout_rate": 0.15
  },
  "training": {
    "n_playout": 800,
    "fpu_reduction": 0.3,
    "grad_clip": 5.0,
    "lr_warmup_steps": 1000
  }
}
```

#### For Fast Training (Limited GPU Memory)
```json
{
  "network": {
    "num_channels": 128,
    "num_res_blocks": 3,
    "use_se": false,
    "dropout_rate": 0.0
  },
  "training": {
    "batch_size": 256,
    "n_playout": 400
  }
}
```

> **GPU Memory Optimization**: Adjust `batch_size`, `num_channels`, and `num_res_blocks` according to your GPU memory. Lower values reduce model size and memory usage.

## References

- Special thanks to [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git) for providing the core codebase.

- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016)
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017)
