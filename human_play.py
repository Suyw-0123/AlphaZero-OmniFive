# -*- coding: utf-8 -*-
"""
human VS AI models with GUI
Click on the board to make your move.

@author: Suyw
"""

from __future__ import print_function
import argparse
import sys
import tkinter as tk
import torch
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from config_loader import load_config, ConfigError
from game_gui import GameGUI, GUIHumanPlayer


class GameController:
    """Controls the game flow with GUI."""
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.app_config = None
        self.board = None
        self.gui = None
        self.human_player = None
        self.ai_player = None
        self.game_running = False
        self.restart_requested = False
        self.quit_requested = False
    
    def load_config(self):
        """Load configuration from file."""
        try:
            self.app_config = load_config(self.config_path)
        except ConfigError as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc
    
    def setup_game(self):
        """Set up the game board and players."""
        self.load_config()
        
        board_cfg = self.app_config.board
        human_cfg = self.app_config.human
        network_cfg = self.app_config.network
        
        width, height = board_cfg.width, board_cfg.height
        n = board_cfg.n_in_row
        model_file = human_cfg.model_file
        use_gpu = human_cfg.use_gpu
        
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but was not detected.")
        
        # Create board
        self.board = Board(width=width, height=height, n_in_row=n)
        
        # Load AI model
        if not model_file:
            raise ValueError("human_play.model_file must be specified in config.json")
        
        best_policy = PolicyValueNet(width, height,
                                     model_file=model_file,
                                     use_gpu=use_gpu,
                                     num_channels=network_cfg.num_channels,
                                     num_res_blocks=network_cfg.num_res_blocks)
        
        self.ai_player = MCTSPlayer(best_policy.policy_value_fn,
                                    c_puct=human_cfg.c_puct,
                                    n_playout=human_cfg.n_playout)
        self.ai_player.is_human = False
        
        # Create GUI
        self.gui = GameGUI(self.board,
                          mode="human",
                          model_file=model_file,
                          on_restart=self.on_restart,
                          on_quit=self.on_quit)
        
        # Create human player
        self.human_player = GUIHumanPlayer(self.gui)
        
        return human_cfg.start_player
    
    def on_restart(self):
        """Handle restart request."""
        self.restart_requested = True
        self.game_running = False
    
    def on_quit(self):
        """Handle quit request."""
        self.game_running = False
        self.restart_requested = False
        self.quit_requested = True
    
    def play_game(self, start_player):
        """Play a single game."""
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        
        self.human_player.set_player_ind(p1)
        self.ai_player.set_player_ind(p2)
        players = {p1: self.human_player, p2: self.ai_player}
        
        self.gui.update_info(p1, p2)
        self.gui.update_board()
        
        self.game_running = True
        self.restart_requested = False
        
        while self.game_running:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            
            # Get move with temp=1e-10 (essentially 0) for AI to play deterministically
            if hasattr(player_in_turn, 'is_human') and not player_in_turn.is_human:
                move = player_in_turn.get_action(self.board, temp=1e-10)
            else:
                move = player_in_turn.get_action(self.board)
            
            if not self.game_running:
                break
            
            self.board.do_move(move)
            
            # Update AI move display
            if hasattr(player_in_turn, 'is_human') and not player_in_turn.is_human:
                self.gui.update_last_move(move, "AI")
            
            self.gui.update_board()
            
            end, winner = self.board.game_end()
            if end:
                self.gui.show_winner(winner, players)
                self.game_running = False
                break
        
        return self.restart_requested
    
    def run(self):
        """Main game loop."""
        try:
            start_player = self.setup_game()
            
            while True:
                restart = self.play_game(start_player)
                if not restart:
                    # Wait for restart or quit
                    self.wait_for_restart_or_quit()
                    if not self.restart_requested:
                        break
                # Reset for new game
                self.gui.move_label.config(text="Last AI move: --")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def wait_for_restart_or_quit(self):
        """Wait for user to click restart or quit after game ends."""
        while not self.restart_requested and not self.quit_requested:
            try:
                self.gui.root.update()
            except tk.TclError:
                # Window was closed
                self.quit_requested = True
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the trained AlphaZero-OmniFive agent with GUI.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    args = parser.parse_args()
    
    controller = GameController(config_path=args.config)
    controller.run()


if __name__ == '__main__':
    main()
