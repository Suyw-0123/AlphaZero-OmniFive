# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
from config_utils import get_section


BOARD_CFG = get_section('board')
PLAY_CFG = get_section('human_play')


def _get(cfg, key, default):
    value = cfg.get(key)
    return default if value is None else value


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
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def _load_numpy_model(model_file, width, height):
    if not model_file:
        raise ValueError("model_file must be provided when using the numpy framework.")
    try:
        policy_param = pickle.load(open(model_file, 'rb'))
    except Exception:
        policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
    return PolicyValueNetNumpy(width, height, policy_param)


def run():
    width = int(_get(BOARD_CFG, 'width', 9))
    height = int(_get(BOARD_CFG, 'height', 9))
    n = int(_get(BOARD_CFG, 'n_in_row', 5))
    play_model_file = _get(PLAY_CFG, 'model_file', 'best_policy.model')
    model_file = None if play_model_file in (None, '') else play_model_file
    start_player = int(_get(PLAY_CFG, 'start_player', 1))
    n_playout = int(_get(PLAY_CFG, 'n_playout', 400))
    c_puct = int(_get(PLAY_CFG, 'c_puct', 5))
    use_gpu = _get(PLAY_CFG, 'use_gpu', True)
    if use_gpu is None:
        use_gpu = True
    framework = str(_get(PLAY_CFG, 'framework', 'pytorch')).lower()
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net according to config
        if framework == 'pytorch':
            best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=bool(use_gpu))
        elif framework == 'numpy':
            best_policy = _load_numpy_model(model_file, width, height)
        else:
            raise ValueError("Unsupported framework '{}'. Use 'pytorch' or 'numpy'.".format(framework))
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=c_puct,
                                 n_playout=n_playout)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=start_player, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
