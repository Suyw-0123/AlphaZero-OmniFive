"""Utility helpers for loading project configuration.

The configuration is stored in an INI-style .cfg file that contains
sections (e.g. [board], [training], [human_play]). This module reads the
file once and performs light type coercion so the rest of the code base can
work with native Python types (int, float, bool) without repeating parsing
logic.
"""
from __future__ import annotations

import configparser
import os
from typing import Any, Dict

_CONFIG_CACHE: Dict[str, Dict[str, Any]] | None = None


def _coerce_value(raw: str) -> Any:
    """Convert a raw string from the config file into a Python value."""
    value = raw.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        # Try integer conversion before float to keep integers precise.
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str = "config.cfg") -> Dict[str, Dict[str, Any]]:
    """Load configuration data from *path*.

    The file is optional: when it is missing an empty dict is returned so the
    calling code can fall back to its built-in defaults.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config = configparser.ConfigParser()
    data: Dict[str, Dict[str, Any]] = {}

    if os.path.exists(path):
        config.read(path)
        for section in config.sections():
            data[section] = {}
            for key, raw_value in config.items(section):
                data[section][key] = _coerce_value(raw_value)
    _CONFIG_CACHE = data
    return data


def get_section(name: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a configuration section by *name* with an optional *default*."""
    config = load_config()
    return config.get(name, {} if default is None else default.copy())
