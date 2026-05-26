from __future__ import annotations

from typing import Any


def train_skrl_ppo(*args: Any, **kwargs: Any) -> Any:
    from oceanscale.training.skrl_trainer import train_skrl_ppo as _train_skrl_ppo

    return _train_skrl_ppo(*args, **kwargs)

__all__ = ["train_skrl_ppo"]
