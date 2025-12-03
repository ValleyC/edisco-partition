"""
CVRP Instance Generation.

Generates random CVRP instances with:
- Depot at index 0
- Customer nodes with demands
- Vehicle capacity constraint
"""

import torch

# Standard capacities for different problem sizes (from GLOP)
CAPACITIES = {
    100: 50.,
    200: 80.,
    500: 100.,
    1000: 200.,
    2000: 300.,
    5000: 300.,
    7000: 300.,
    10000: 400.,
}


def generate_instance(n_customers, device='cuda'):
    """
    Generate a random CVRP instance.

    Args:
        n_customers: Number of customer nodes (excluding depot)
        device: Device to place tensors on

    Returns:
        coords: (n_customers + 1, 2) - coordinates including depot at index 0
        demand: (n_customers + 1,) - demands, depot has demand 0
        capacity: Vehicle capacity
    """
    n_nodes = n_customers + 1  # +1 for depot

    # Random coordinates in [0, 1]^2
    coords = torch.rand(n_nodes, 2, device=device)

    # Random demands in {1, ..., 9}
    demand = torch.randint(1, 10, (n_nodes,), device=device, dtype=torch.float)
    demand[0] = 0  # Depot has no demand

    # Get capacity for this problem size
    capacity = CAPACITIES.get(n_customers, 50. + n_customers * 0.1)

    return coords, demand, capacity


def generate_batch(n_customers, batch_size, device='cuda'):
    """
    Generate a batch of CVRP instances.

    Args:
        n_customers: Number of customers per instance
        batch_size: Number of instances
        device: Device

    Returns:
        List of (coords, demand, capacity) tuples
    """
    instances = []
    for _ in range(batch_size):
        instances.append(generate_instance(n_customers, device))
    return instances


def load_instance_from_file(filepath):
    """
    Load a CVRP instance from file.

    Supports simple format:
    - First line: n_customers capacity
    - Second line: depot_x depot_y
    - Following lines: customer_x customer_y demand
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip().split()
    n_customers = int(first_line[0])
    capacity = float(first_line[1])

    coords = []
    demands = []

    # Depot
    depot_line = lines[1].strip().split()
    coords.append([float(depot_line[0]), float(depot_line[1])])
    demands.append(0.0)

    # Customers
    for i in range(2, 2 + n_customers):
        parts = lines[i].strip().split()
        coords.append([float(parts[0]), float(parts[1])])
        demands.append(float(parts[2]))

    coords = torch.tensor(coords)
    demands = torch.tensor(demands)

    return coords, demands, capacity
