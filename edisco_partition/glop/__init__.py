"""
GLOP modules for Reviser-based TSP solving.

This package contains adapted code from GLOP (Global and Local Optimization for CVRP)
for using pretrained Reviser models during partition network training.
"""
from .utils.functions import load_model, reconnect, load_problem
