# Configuration Presets

This directory contains preset configurations optimized for different scenarios. Use these as starting points for your training.

## Available Presets

### 1. `small_board_optimized.json` - For 8x8 Boards
**Best for**: Quick training and testing on smaller boards

- Board: 8x8
- Network: 128 channels, 4 residual blocks with SE attention
- MCTS: 400 playouts per move
- Features: SE blocks enabled, moderate dropout (0.1)
- Training time: Fast convergence

**Usage**:
```bash
python train.py --config configs/small_board_optimized.json
```

### 2. `large_board_optimized.json` - For 15x15 Boards
**Best for**: Standard Gomoku on larger boards

- Board: 15x15
- Network: 256 channels, 6 residual blocks with SE attention
- MCTS: 1000 playouts per move
- Features: SE blocks enabled, higher dropout (0.15)
- Training time: Moderate, requires more GPU memory

**Usage**:
```bash
python train.py --config configs/large_board_optimized.json
```

### 3. `fast_training.json` - Quick Training Mode
**Best for**: Limited GPU memory or quick experiments

- Board: 11x11 (standard)
- Network: 128 channels, 3 residual blocks, NO SE blocks
- MCTS: 400 playouts per move
- Features: Minimal complexity, faster iterations
- Training time: Very fast, suitable for testing

**Usage**:
```bash
python train.py --config configs/fast_training.json
```

### 4. `maximum_strength.json` - Maximum AI Strength
**Best for**: Creating the strongest possible AI (requires powerful GPU)

- Board: 11x11 (standard)
- Network: 384 channels, 8 residual blocks with SE attention
- MCTS: 1600 playouts per move
- Features: All optimizations enabled, maximum model capacity
- Training time: Slow, requires high-end GPU (8GB+ VRAM)

**Usage**:
```bash
python train.py --config configs/maximum_strength.json
```

## Key Optimization Parameters

### Network Architecture
- **SE Blocks** (`use_se`): Channel attention mechanism for better feature learning
- **Dropout** (`dropout_rate`): Regularization to prevent overfitting
- **Channels & Blocks**: More = stronger but slower

### MCTS Parameters
- **FPU Reduction** (`fpu_reduction`): Controls exploration of unvisited nodes
  - Lower (0.2): More aggressive, faster decisions
  - Higher (0.35): More thorough exploration
- **Playouts** (`n_playout`): More = stronger but slower

### Training Stability
- **Gradient Clipping** (`grad_clip`): Always set to 5.0 for stability
- **LR Warmup** (`lr_warmup_steps`): Stabilizes early training
  - Small boards: 500 steps
  - Large boards: 1000-1500 steps

## Customization Tips

1. **Limited GPU Memory?**
   - Reduce `batch_size` (512 → 256)
   - Reduce `num_channels` (256 → 128)
   - Disable SE blocks (`use_se: false`)

2. **Want Faster Training?**
   - Reduce `n_playout` (800 → 400)
   - Reduce `num_res_blocks` (6 → 3-4)
   - Increase `play_batch_size` (parallel games)

3. **Want Stronger AI?**
   - Increase `num_channels` (256 → 384)
   - Increase `num_res_blocks` (4 → 6-8)
   - Increase `n_playout` (800 → 1600)
   - Enable all optimizations

4. **Prevent Overfitting?**
   - Increase `dropout_rate` (0.1 → 0.2)
   - Increase `buffer_size` (more diverse data)
   - Increase `l2_const` (1e-4 → 5e-4)

## Performance Comparison

| Preset | Training Speed | AI Strength | GPU Memory | Convergence |
|--------|---------------|-------------|------------|-------------|
| Small Board | ⚡⚡⚡⚡ Fast | ⭐⭐⭐ Good | 2-3 GB | Quick |
| Large Board | ⚡⚡⚡ Moderate | ⭐⭐⭐⭐ Very Good | 4-6 GB | Moderate |
| Fast Training | ⚡⚡⚡⚡⚡ Very Fast | ⭐⭐ Basic | 1-2 GB | Very Quick |
| Maximum Strength | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | 8+ GB | Slow |

## Notes

- All presets use gradient clipping and learning rate warmup for stability
- The `use_cosine_annealing` parameter is prepared for future implementation
- Adjust `check_freq` based on how often you want to evaluate the model
- Higher `pure_mcts_playout_num` makes evaluation more rigorous but slower
