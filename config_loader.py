"""Typed configuration loader for AlphaZero-OmniFive.

The configuration is stored in a simple JSON document.  This module exposes
immutable dataclasses for each configuration section and validates the
parameters at load time to keep the rest of the codebase lean and clean.

@author: Suyw
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_CONFIG_PATH = Path("config.json")


class ConfigError(RuntimeError):
    """Raised when the configuration file contains invalid data."""


def _ensure_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer (got {value}).")
    return value


def _ensure_between(value: int, low: int, high: int, name: str) -> int:
    if not low <= value <= high:
        raise ConfigError(f"{name} must be in range [{low}, {high}] (got {value}).")
    return value


@dataclass(frozen=True)
class BoardConfig:
    width: int = 8
    height: int = 8
    n_in_row: int = 5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BoardConfig":
        default = cls()
        data = data or {}
        width = _ensure_positive(int(data.get("width", default.width)), "board.width")
        height = _ensure_positive(int(data.get("height", default.height)), "board.height")
        max_in_row = min(width, height)
        n_in_row = int(data.get("n_in_row", default.n_in_row))
        n_in_row = _ensure_between(n_in_row, 2, max_in_row, "board.n_in_row")
        return cls(width=width, height=height, n_in_row=n_in_row)


@dataclass(frozen=True)
class NetworkConfig:
    num_channels: int = 128
    num_res_blocks: int = 6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NetworkConfig":
        default = cls()
        data = data or {}
        num_channels = _ensure_positive(
            int(data.get("num_channels", default.num_channels)), "network.num_channels"
        )
        num_res_blocks = _ensure_positive(
            int(data.get("num_res_blocks", default.num_res_blocks)), "network.num_res_blocks"
        )
        return cls(num_channels=num_channels, num_res_blocks=num_res_blocks)


@dataclass(frozen=True)
class TrainingConfig:
    learn_rate: float = 2e-3
    lr_multiplier: float = 1.0
    temp: float = 1.0
    n_playout: int = 900
    c_puct: int = 5
    buffer_size: int = 20000
    batch_size: int = 768
    play_batch_size: int = 2
    epochs: int = 8
    kl_targ: float = 0.02
    check_freq: int = 30
    game_batch_num: int = 2500
    pure_mcts_playout_num: int = 2000
    use_gpu: bool = True
    init_model: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrainingConfig":
        default = cls()
        data = data or {}
        learn_rate = float(data.get("learn_rate", default.learn_rate))
        if learn_rate <= 0:
            raise ConfigError(f"training.learn_rate must be > 0 (got {learn_rate}).")
        lr_multiplier = float(data.get("lr_multiplier", default.lr_multiplier))
        if lr_multiplier <= 0:
            raise ConfigError(f"training.lr_multiplier must be > 0 (got {lr_multiplier}).")
        temp = float(data.get("temp", default.temp))
        if temp <= 0:
            raise ConfigError(f"training.temp must be > 0 (got {temp}).")
        n_playout = _ensure_positive(int(data.get("n_playout", default.n_playout)), "training.n_playout")
        c_puct = _ensure_positive(int(data.get("c_puct", default.c_puct)), "training.c_puct")
        buffer_size = _ensure_positive(int(data.get("buffer_size", default.buffer_size)), "training.buffer_size")
        batch_size = _ensure_positive(int(data.get("batch_size", default.batch_size)), "training.batch_size")
        play_batch_size = _ensure_positive(int(data.get("play_batch_size", default.play_batch_size)), "training.play_batch_size")
        epochs = _ensure_positive(int(data.get("epochs", default.epochs)), "training.epochs")
        kl_targ = float(data.get("kl_targ", default.kl_targ))
        if kl_targ <= 0:
            raise ConfigError(f"training.kl_targ must be > 0 (got {kl_targ}).")
        check_freq = _ensure_positive(int(data.get("check_freq", default.check_freq)), "training.check_freq")
        game_batch_num = _ensure_positive(int(data.get("game_batch_num", default.game_batch_num)), "training.game_batch_num")
        pure_mcts_playout_num = _ensure_positive(int(data.get("pure_mcts_playout_num", default.pure_mcts_playout_num)), "training.pure_mcts_playout_num")
        use_gpu = bool(data.get("use_gpu", default.use_gpu))
        init_model_raw = data.get("init_model", default.init_model or "")
        init_model = str(init_model_raw).strip() or None
        return cls(
            learn_rate=learn_rate,
            lr_multiplier=lr_multiplier,
            temp=temp,
            n_playout=n_playout,
            c_puct=c_puct,
            buffer_size=buffer_size,
            batch_size=batch_size,
            play_batch_size=play_batch_size,
            epochs=epochs,
            kl_targ=kl_targ,
            check_freq=check_freq,
            game_batch_num=game_batch_num,
            pure_mcts_playout_num=pure_mcts_playout_num,
            use_gpu=use_gpu,
            init_model=init_model,
        )


@dataclass(frozen=True)
class HumanPlayConfig:
    model_file: str = "best_policy.model"
    start_player: int = 1
    n_playout: int = 400
    c_puct: int = 5
    use_gpu: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "HumanPlayConfig":
        default = cls()
        data = data or {}
        model_file_raw = data.get("model_file", default.model_file)
        model_file = str(model_file_raw).strip()
        if not model_file:
            raise ConfigError("human_play.model_file cannot be empty.")
        start_player = int(data.get("start_player", default.start_player))
        start_player = _ensure_between(start_player, 0, 1, "human_play.start_player")
        n_playout = _ensure_positive(int(data.get("n_playout", default.n_playout)), "human_play.n_playout")
        c_puct = _ensure_positive(int(data.get("c_puct", default.c_puct)), "human_play.c_puct")
        use_gpu = bool(data.get("use_gpu", default.use_gpu))
        return cls(
            model_file=model_file,
            start_player=start_player,
            n_playout=n_playout,
            c_puct=c_puct,
            use_gpu=use_gpu,
        )


@dataclass(frozen=True)
class AppConfig:
    board: BoardConfig
    network: NetworkConfig
    training: TrainingConfig
    human: HumanPlayConfig


def _load_raw_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Failed to parse JSON from {path}: {exc}") from exc


@lru_cache(maxsize=None)
def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate configuration from *path*.

    The result is cached to avoid re-parsing in long-running processes.
    """

    config_path = Path(path)
    raw = _load_raw_config(config_path)
    board = BoardConfig.from_dict(raw.get("board"))
    network = NetworkConfig.from_dict(raw.get("network"))
    training = TrainingConfig.from_dict(raw.get("training"))
    human = HumanPlayConfig.from_dict(raw.get("human_play"))
    return AppConfig(board=board, network=network, training=training, human=human)


__all__ = [
    "AppConfig",
    "BoardConfig",
    "ConfigError",
    "HumanPlayConfig",
    "NetworkConfig",
    "TrainingConfig",
    "load_config",
]
