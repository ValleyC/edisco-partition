"""Data utilities for CVRP instance generation."""

from .instance import generate_instance, CAPACITIES
from .graph import build_graph

__all__ = ["generate_instance", "build_graph", "CAPACITIES"]
