# -*- coding: utf-8 -*-
"""
Simple Tkinter GUI for Gomoku game display.
Supports both human_play and battle modes.

@author: Suyw
"""

import tkinter as tk
from tkinter import messagebox


class GameGUI:
    """
    A simple Tkinter-based GUI for displaying the Gomoku board.
    """

    def __init__(self, board, mode="human", player1_name="Player 1", player2_name="Player 2",
                 model_file="", on_restart=None, on_quit=None):
        self.board = board
        self.mode = mode
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.model_file = model_file
        self.on_restart = on_restart
        self.on_quit = on_quit
        
        self.cell_size = 45
        self.margin = 35
        self.width = board.width
        self.height = board.height
        
        self.canvas_width = self.width * self.cell_size + 2 * self.margin
        self.canvas_height = self.height * self.cell_size + 2 * self.margin
        
        self.root = None
        self.canvas = None
        self.info_label = None
        self.move_label = None
        self.human_move = None
        self.waiting_for_human = False
        
        self.player1_piece = None
        self.player2_piece = None
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Set up the Tkinter GUI components."""
        self.root = tk.Tk()
        self.root.title("AlphaZero-OmniFive")
        self.root.resizable(False, False)
        
        # Top frame for info
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)
        
        # Player info display
        if self.mode == "human":
            info_text = f"Human: ● (Black)    Model [{self.model_file}]: ○ (White)"
        else:
            info_text = f"Model [{self.model_file}]: ● (Black)    Pure MCTS: ○ (White)"
        
        self.info_label = tk.Label(top_frame, text=info_text, font=("Arial", 11))
        self.info_label.pack()
        
        # Last move display
        self.move_label = tk.Label(top_frame, text="Last AI move: --", font=("Arial", 10))
        self.move_label.pack(pady=5)
        
        # Canvas for the board
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="#DEB887")
        self.canvas.pack()
        
        # Bind click event for human play
        if self.mode == "human":
            self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Bottom frame for buttons
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10)
        
        restart_btn = tk.Button(bottom_frame, text="Restart", font=("Arial", 10), width=10,
                                command=self._on_restart_click)
        restart_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = tk.Button(bottom_frame, text="Quit", font=("Arial", 10), width=10,
                             command=self._on_quit_click)
        quit_btn.pack(side=tk.LEFT, padx=10)
        
        self._draw_board()
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit_click)
    
    def _draw_board(self):
        """Draw the game board grid and coordinates."""
        self.canvas.delete("all")
        
        for i in range(self.width):
            x = self.margin + i * self.cell_size + self.cell_size // 2
            y1 = self.margin + self.cell_size // 2
            y2 = self.margin + (self.height - 1) * self.cell_size + self.cell_size // 2
            self.canvas.create_line(x, y1, x, y2, fill="black")
            self.canvas.create_text(x, self.margin // 2, text=str(i), font=("Arial", 9))
        
        for i in range(self.height):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            x1 = self.margin + self.cell_size // 2
            x2 = self.margin + (self.width - 1) * self.cell_size + self.cell_size // 2
            self.canvas.create_line(x1, y, x2, y, fill="black")
            row_label = self.height - 1 - i
            self.canvas.create_text(self.margin // 2, y, text=str(row_label), font=("Arial", 9))
        
        self._draw_pieces()
    
    def _draw_pieces(self):
        """Draw all pieces on the board."""
        for move, player in self.board.states.items():
            self._draw_piece(move, player)
        
        # Highlight last move if exists
        if hasattr(self.board, 'last_move') and self.board.last_move != -1:
            self._highlight_last_move(self.board.last_move)
    
    def _draw_piece(self, move, player):
        """Draw a single piece on the board."""
        h, w = self.board.move_to_location(move)
        canvas_row = self.height - 1 - h
        x = self.margin + w * self.cell_size + self.cell_size // 2
        y = self.margin + canvas_row * self.cell_size + self.cell_size // 2
        
        radius = self.cell_size // 2 - 4
        
        if player == 1:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill="black", outline="black", tags="piece")
        else:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill="white", outline="black", width=2, tags="piece")
    
    def _highlight_last_move(self, move):
        """Highlight the last move with a small red marker."""
        h, w = self.board.move_to_location(move)
        canvas_row = self.height - 1 - h
        x = self.margin + w * self.cell_size + self.cell_size // 2
        y = self.margin + canvas_row * self.cell_size + self.cell_size // 2
        
        marker_size = 5
        self.canvas.create_rectangle(x - marker_size, y - marker_size,
                                     x + marker_size, y + marker_size,
                                     fill="red", outline="red", tags="highlight")
    
    def _on_canvas_click(self, event):
        """Handle canvas click for human move input."""
        if not self.waiting_for_human:
            return
        
        col = (event.x - self.margin) // self.cell_size
        canvas_row = (event.y - self.margin) // self.cell_size
        row = self.height - 1 - canvas_row
        
        if 0 <= col < self.width and 0 <= row < self.height:
            move = self.board.location_to_move([row, col])
            if move in self.board.availables:
                self.human_move = move
                self.waiting_for_human = False
    
    def _on_restart_click(self):
        """Handle restart button click."""
        self.waiting_for_human = False  # Break out of waiting loop
        self.human_move = None
        if self.on_restart:
            self.on_restart()
    
    def _on_quit_click(self):
        """Handle quit button click."""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.waiting_for_human = False  # Break out of waiting loop
            if self.on_quit:
                self.on_quit()
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
            import sys
            sys.exit(0)
    
    def update_board(self):
        """Update the board display."""
        self._draw_board()
        self.root.update()
    
    def update_info(self, player1_piece, player2_piece):
        """Update player piece assignment info."""
        self.player1_piece = player1_piece
        self.player2_piece = player2_piece
        
        p1_symbol = "● (Black)" if player1_piece == 1 else "○ (White)"
        p2_symbol = "● (Black)" if player2_piece == 1 else "○ (White)"
        
        if self.mode == "human":
            info_text = f"Human: {p1_symbol}    Model [{self.model_file}]: {p2_symbol}"
        else:
            info_text = f"Model [{self.model_file}]: {p1_symbol}    Pure MCTS: {p2_symbol}"
        
        self.info_label.config(text=info_text)
    
    def update_last_move(self, move, player_name="AI"):
        """Update the last move display."""
        if move is not None and move != -1:
            h, w = self.board.move_to_location(move)
            self.move_label.config(text=f"Last {player_name} move: ({h}, {w})")
        else:
            self.move_label.config(text=f"Last {player_name} move: --")
    
    def get_human_move(self):
        """Wait for and return human move from GUI click."""
        self.human_move = None
        self.waiting_for_human = True
        
        while self.waiting_for_human:
            self.root.update()
        
        return self.human_move
    
    def show_winner(self, winner, players):
        """Display the game result."""
        if winner == -1:
            result = "Game ended in a Tie!"
        else:
            # Determine winner name based on mode
            winner_player = players.get(winner)
            if self.mode == "human":
                # Human play mode: show "Human" or model name
                if hasattr(winner_player, 'is_human') and winner_player.is_human:
                    winner_name = "Human"
                else:
                    winner_name = f"Model [{self.model_file}]"
            else:
                # Battle mode: show model name or "Pure MCTS"
                if hasattr(winner_player, 'name'):
                    winner_name = winner_player.name
                elif self.player1_piece == winner:
                    winner_name = f"Model [{self.model_file}]"
                else:
                    winner_name = "Pure MCTS"
            result = f"Game Over! Winner: {winner_name}"
        
        self.move_label.config(text=result)
        messagebox.showinfo("Game Over", result)
    
    def mainloop(self):
        """Start the main loop (for keeping window open after game)."""
        self.root.mainloop()


class GUIHumanPlayer:
    """Human player that gets moves from GUI."""
    
    def __init__(self, gui):
        self.gui = gui
        self.player = None
        self.is_human = True
    
    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board):
        return self.gui.get_human_move()
    
    def __str__(self):
        return f"Human {self.player}"
