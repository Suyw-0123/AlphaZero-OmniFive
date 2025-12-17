# -*- coding: utf-8 -*-
"""
Dynamic Training Parameters for AlphaZero
Implements adaptive adjustment of training hyperparameters during training.

@author: Suyw
"""

import numpy as np


class DynamicTrainingParams:
    """
    Dynamically adjust training parameters based on training progress.
    
    This class implements techniques from AlphaGo Zero paper and community best practices:
    1. Dynamic playout budget: Start with fewer MCTS playouts, increase as network strengthens
    2. Adaptive c_puct: High exploration early, shift to exploitation later
    3. Temperature annealing: High randomness early, deterministic later
    """
    
    def __init__(self, config=None):
        """
        Initialize dynamic training parameters.
        
        Args:
            config (dict): Configuration dictionary containing:
                - initial_n_playout: Starting MCTS simulation count (default: 400)
                - final_n_playout: Final MCTS simulation count (default: 1200)
                - initial_c_puct: Starting exploration constant (default: 5.0)
                - final_c_puct: Final exploration constant (default: 2.0)
                - initial_temp: Starting temperature (default: 1.0)
                - final_temp: Final temperature (default: 0.1)
                - warmup_batches: Batches for playout warmup (default: 500)
                - anneal_batches: Batches for parameter annealing (default: 1000)
                - enable_dynamic_playout: Enable dynamic playout (default: True)
                - enable_dynamic_c_puct: Enable dynamic c_puct (default: True)
                - enable_dynamic_temp: Enable dynamic temperature (default: True)
        """
        if config is None:
            config = {}
        
        # Playout budget parameters
        self.initial_n_playout = config.get('initial_n_playout', 400)
        self.final_n_playout = config.get('final_n_playout', 1200)
        
        # Exploration constant parameters
        self.initial_c_puct = config.get('initial_c_puct', 5.0)
        self.final_c_puct = config.get('final_c_puct', 2.0)
        
        # Temperature parameters
        self.initial_temp = config.get('initial_temp', 1.0)
        self.final_temp = config.get('final_temp', 0.1)
        
        # Schedule parameters
        self.warmup_batches = config.get('warmup_batches', 500)
        self.anneal_batches = config.get('anneal_batches', 1000)
        
        # Enable/disable flags
        self.enable_dynamic_playout = config.get('enable_dynamic_playout', True)
        self.enable_dynamic_c_puct = config.get('enable_dynamic_c_puct', True)
        self.enable_dynamic_temp = config.get('enable_dynamic_temp', True)
    
    def get_n_playout(self, batch_i):
        """
        Get dynamic MCTS playout count based on training progress.
        
        Rationale:
        - Early training: Network is weak, MCTS benefit is limited → fewer playouts for speed
        - Late training: Network is strong, MCTS can find better moves → more playouts for quality
        
        Args:
            batch_i (int): Current training batch index
            
        Returns:
            int: Number of MCTS playouts per move
        """
        if not self.enable_dynamic_playout:
            return self.final_n_playout
        
        if batch_i < self.warmup_batches:
            # Linear interpolation during warmup
            progress = batch_i / self.warmup_batches
            n_playout = self.initial_n_playout + \
                       (self.final_n_playout - self.initial_n_playout) * progress
            return int(n_playout)
        
        return self.final_n_playout
    
    def get_c_puct(self, batch_i):
        """
        Get dynamic exploration constant (c_puct) based on training progress.
        
        Rationale:
        - Early training: Need high exploration to discover new strategies
        - Late training: Network is accurate, can exploit known good moves
        
        PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + n(s,a))
                                ↑
                      Controls exploration weight
        
        Args:
            batch_i (int): Current training batch index
            
        Returns:
            float: Exploration constant
        """
        if not self.enable_dynamic_c_puct:
            return self.initial_c_puct
        
        if batch_i < self.anneal_batches:
            # Linear decay during annealing
            progress = batch_i / self.anneal_batches
            c_puct = self.initial_c_puct - \
                    (self.initial_c_puct - self.final_c_puct) * progress
            return c_puct
        
        return self.final_c_puct
    
    def get_temperature(self, batch_i, move_step=0):
        """
        Get dynamic temperature for action sampling.
        
        Rationale:
        - High temperature: More random moves → diverse training data
        - Low temperature: Near-greedy moves → sharper policy
        
        Temperature controls action selection:
        P(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)
        
        τ = 1.0 → Sample proportional to visit counts
        τ → 0   → Select most-visited action (greedy)
        
        Args:
            batch_i (int): Current training batch index
            move_step (int): Current move number in game (optional)
            
        Returns:
            float: Temperature parameter
        """
        if not self.enable_dynamic_temp:
            return self.initial_temp
        
        # Training progress annealing
        if batch_i < self.anneal_batches:
            progress = batch_i / self.anneal_batches
            base_temp = self.initial_temp - \
                       (self.initial_temp - self.final_temp) * progress
        else:
            base_temp = self.final_temp
        
        # Optional: In-game annealing (AlphaGo Zero style)
        # Keep high temperature for first N moves, then reduce
        if move_step > 30:
            # After opening phase, use lower temperature
            return max(base_temp * 0.1, 0.01)
        
        return base_temp
    
    def get_all_params(self, batch_i):
        """
        Get all dynamic parameters for current batch.
        
        Args:
            batch_i (int): Current training batch index
            
        Returns:
            dict: Dictionary containing all dynamic parameters
        """
        return {
            'n_playout': self.get_n_playout(batch_i),
            'c_puct': self.get_c_puct(batch_i),
            'temperature': self.get_temperature(batch_i)
        }
    
    def log_params(self, batch_i, writer=None):
        """
        Log current parameters to console and optionally TensorBoard.
        
        Args:
            batch_i (int): Current training batch index
            writer: TensorBoard SummaryWriter (optional)
        """
        params = self.get_all_params(batch_i)
        
        print(f"[Dynamic Params @ batch {batch_i}] "
              f"n_playout={params['n_playout']}, "
              f"c_puct={params['c_puct']:.2f}, "
              f"temp={params['temperature']:.3f}")
        
        # Optional TensorBoard logging
        if writer is not None:
            writer.add_scalar('DynamicParams/n_playout', params['n_playout'], batch_i)
            writer.add_scalar('DynamicParams/c_puct', params['c_puct'], batch_i)
            writer.add_scalar('DynamicParams/temperature', params['temperature'], batch_i)
    
    def __str__(self):
        """String representation of configuration"""
        return (f"DynamicTrainingParams(\n"
                f"  n_playout: {self.initial_n_playout} → {self.final_n_playout} "
                f"(warmup: {self.warmup_batches} batches)\n"
                f"  c_puct: {self.initial_c_puct} → {self.final_c_puct} "
                f"(anneal: {self.anneal_batches} batches)\n"
                f"  temp: {self.initial_temp} → {self.final_temp} "
                f"(anneal: {self.anneal_batches} batches)\n"
                f"  enabled: playout={self.enable_dynamic_playout}, "
                f"c_puct={self.enable_dynamic_c_puct}, "
                f"temp={self.enable_dynamic_temp}\n"
                f")")
