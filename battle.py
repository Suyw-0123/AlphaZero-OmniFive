# -*- coding: utf-8 -*-
"""
A script to pit the trained AlphaZero model against a pure MCTS player.

@author: Suyw (modified for AI vs AI battle)
"""

from __future__ import print_function
import argparse
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from config_loader import load_config, ConfigError


def run(config_path="config.json"):
    """
    Run a game between the trained AlphaZero model and a pure MCTS player.
    """
    try:
        app_config = load_config(config_path)
    except ConfigError as exc:
        raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    board_cfg = app_config.board
    human_cfg = app_config.human  # We'll use human_play config for the AlphaZero model
    width, height = board_cfg.width, board_cfg.height
    n = board_cfg.n_in_row
    model_file = human_cfg.model_file
    use_gpu = human_cfg.use_gpu

    try:
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but was not detected. Set human_play.use_gpu=false in config.json to run on CPU.")
        
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### AI vs AI ###################
        if not model_file:
            raise ValueError("human_play.model_file must point to a PyTorch checkpoint saved by train.py")

        # Player 1: The trained AlphaZero model
        network_cfg = app_config.network
        alpha_zero_policy = PolicyValueNet(width, height,
                                           model_file=model_file,
                                           use_gpu=use_gpu,
                                           num_channels=network_cfg.num_channels,
                                           num_res_blocks=network_cfg.num_res_blocks)
        
        alpha_zero_player = MCTSPlayer(alpha_zero_policy.policy_value_fn,
                                       c_puct=human_cfg.c_puct,
                                       n_playout=human_cfg.n_playout,
                                       is_selfplay=0)
        print(f"AlphaZero Player loaded from {model_file} with n_playout={human_cfg.n_playout}")


        # Player 2: The pure MCTS player
        # You can adjust c_puct and n_playout for the pure MCTS player here.
        pure_mcts_playout = 4000 # Making it reasonably strong
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout)
        print(f"Pure MCTS Player created with n_playout={pure_mcts_playout}")


        # Set start_player=0 for AlphaZero to go first, or 1 for pure MCTS to go first.
        # is_shown=1 will print the board state after each move.
        game.start_play(alpha_zero_player, 
                        pure_mcts_player, 
                        start_player=0, 
                        is_shown=1)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file '{model_file}' not found. Run train.py to produce a PyTorch checkpoint before starting."
        )
    except (RuntimeError) as exc:
        raise RuntimeError(
            f"An error occurred: {exc}"
        ) from exc
    except KeyboardInterrupt:
        print('\n\rquit')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a battle between the trained AlphaZero-OmniFive agent and a pure MCTS agent.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == '__main__':
    main()
