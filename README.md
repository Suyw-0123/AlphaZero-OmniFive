# AlphaZero-OmniFive

AlphaZero-OmniFive applies the AlphaZero algorithm to Gomoku (Five in a Row), training a policy-value network purely through self-play data combined with Monte Carlo Tree Search (MCTS) for decision-making. Since Gomoku's state space is much smaller than Go or Chess, a competitive AI can be trained in just a few hours on a PC with a CUDA-enabled GPU.

This repo is based on [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git),And make the following modifications:
- Changed the network architecture from CNN to ResNet
- Optimized MCTS and self-play modules by leveraging PyTorch CUDA acceleration
- Tuned training parameters specifically for large boards of size 9x9 and above
- Added the models trained using this parameter
- Added a new config.json file for centralized parameter management
- Added GUI based on tkinter

#### Differences Between AlphaGo and AlphaGo Zero

- **AlphaGo**: Combines expert game records, hand-crafted features, and move prediction with MCTS, further enhanced through self-play.
- **AlphaGo Zero**: Starts from scratch using only game rules for self-play, employs residual convolutional networks to output both policy and value simultaneously with MCTS; abandons hand-crafted features and human game records for a simpler architecture with more efficient training and inference, surpassing AlphaGo in strength.

![playout400](gomoku_demo.gif)

#### System Requirements (The version I'm using)

- Python >= 3.13
- PyTorch >= 2.9 (CUDA 12.8 required)
- numpy >= 2.2

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
| `num_channels` | Number of feature channels in residual blocks. Higher values increase model capacity. |
| `num_res_blocks` | Number of residual blocks in the tower. More blocks enable deeper feature extraction. |

### Training Configuration

| Parameter | Description |
| --- | --- |
| `learn_rate` | Initial Adam learning rate. Dynamically scaled by `lr_multiplier` based on KL divergence. |
| `lr_multiplier` | Multiplicatively adjusted when KL exceeds or falls below thresholds, controlling learning rate decay or recovery. |
| `temp` | Temperature during self-play, controlling move exploration; can be lowered later to reduce randomness. |
| `n_playout` | Number of MCTS simulations per move. Higher values increase strength but also inference time. |
| `c_puct` | MCTS exploration coefficient, balancing high visit counts and high-scoring nodes. |
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


### Human Play Configuration

| Parameter | Description |
| --- | --- |
| `model_file` | Path to the model file used for human vs AI games. |
| `start_player` | Set to 0 for human first, 1 for AI first. |
| `n_playout` | Number of MCTS simulations per move for the AI during human play. |
| `c_puct` | MCTS exploration coefficient for human play. |
| `use_gpu` | Whether to use GPU acceleration for inference during human play. |

> **GPU Memory Optimization**: Adjust `batch_size`, `num_channels`, and `num_res_blocks` according to your GPU memory. Lower values reduce model size and memory usage.

### Battle Mode: `battle.py`

To evaluate the strength of your trained model, you can pit it against a pure MCTS (Monte Carlo Tree Search) opponent using the `battle.py` script.

#### How to Run

```bash
python battle.py
```

This script will:
1.  Load the trained model specified in `config.json` under `human_play.model_file`.
2.  Create a pure MCTS player to act as the opponent.
3.  Start a game and print the board state to the console after each move.

#### Configuration

You can adjust the battle parameters directly within the `battle.py` file:

- **Opponent Strength**: Modify the `pure_mcts_playout` variable to increase or decrease the thinking time and strength of the pure MCTS player.
- **First Move**: Change the `start_player` argument in the `game.start_play()` function call. Set it to `0` for your trained model to go first, or `1` for the pure MCTS player to go first.


## References

- Special thanks to [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git) for providing the core codebase.

- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016)
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017)
