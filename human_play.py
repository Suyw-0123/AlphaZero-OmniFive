# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import argparse
import pickle
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure  # noqa: F401
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from config_loader import load_config, ConfigError


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(config_path="config.json"):
    try:
        app_config = load_config(config_path)
    except ConfigError as exc:
        raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    board_cfg = app_config.board
    human_cfg = app_config.human
    width, height = board_cfg.width, board_cfg.height
    n = board_cfg.n_in_row
    model_file = human_cfg.model_file
    use_gpu = human_cfg.use_gpu
    try:
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but was not detected. Set human_play.use_gpu=false in config.json to run on CPU.")
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        if not model_file:
            raise ValueError("human_play.model_file must point to a PyTorch checkpoint saved by train.py")

        network_cfg = app_config.network
        best_policy = PolicyValueNet(width, height,
                                     model_file=model_file,
                                     use_gpu=use_gpu,
                                     num_channels=network_cfg.num_channels,
                                     num_res_blocks=network_cfg.num_res_blocks)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=human_cfg.c_puct,
                                 n_playout=human_cfg.n_playout)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=human_cfg.start_player, is_shown=1, temp=human_cfg.temp)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file '{model_file}' not found. Run train.py to produce a PyTorch checkpoint before starting human_play."
        )
    except (RuntimeError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            "Failed to load PyTorch model. Ensure the checkpoint was produced by train.py (PyTorch) rather than the legacy Theano model."
        ) from exc
    except KeyboardInterrupt:
        print('\n\rquit')


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the trained AlphaZero-OmniFive agent.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == '__main__':
    main()
