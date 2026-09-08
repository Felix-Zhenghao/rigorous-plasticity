"""Shared tensor and module types for architecture-specific method adapters."""
from typing import TypeAlias

from torch import nn

# Incoming transform, intervening normalization, and outgoing transform for a unit site.
RecyclingSite: TypeAlias = tuple[nn.Linear | nn.Conv2d, nn.Module, nn.Linear | nn.Conv2d]
