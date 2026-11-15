"""Упрощённый RND (Random Network Distillation) для MiniGrid.

Одна value-голова PPO, RND живёт во wrapper'е и добавляет
β · r_int к внешней награде. См. README_RND.md.
"""
from .running_mean_std import RunningMeanStd
from .rnd_module import RNDModule
from .wrapper import IntrinsicRewardWrapper

__all__ = ["RunningMeanStd", "RNDModule", "IntrinsicRewardWrapper"]