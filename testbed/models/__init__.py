"""Ordinary networks and their reproducible initialization recipes."""
from typing import TypeAlias

from .mlp import MLP
from .resnet_18 import ResNet18
from .vit import ViT

Network: TypeAlias = MLP | ResNet18 | ViT

__all__ = ["MLP", "Network", "ResNet18", "ViT"]
