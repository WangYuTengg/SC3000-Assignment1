# Import necessary libraries and packages
import json
import heapq
import math


def load_json_file(filename):
    """
    Function to load json files into a dictionary
    Parameters:
        filename: name of the json file
    """
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def combine_graphs(G, Dist, Cost):
    """
    Function to combine the graphs into one graph
    Parameters:
        G: graph of the nodes (adjacency list)
        Dist: graph of the distances
        Cost: graph of the costs
    """
    combined_graph = {}

    for node, neighbors in G.items():
        combined_graph[node] = []
        for neighbor in neighbors:
            edge_key = f"{node},{neighbor}"
            edge_data = {
                "node": neighbor,
                "distance": Dist.get(
                    edge_key, float("inf")
                ),  # Using inf as a default if no distance is provided
                "cost": Cost.get(
                    edge_key, float("inf")
                ),  # Using inf as a default if no cost is provided
            }
            combined_graph[node].append(edge_data)

    return combined_graph


def dijkstra(graph, start, target):
    """
    Function to find the shortest path from start to target node (task 1)
    Parameters:
        graph: combined graph of the nodes, distances and costs
        start: start node
        target: target node
    """
    # Initialize distances with infinite values and predecessors to None
    distances = {node: float("infinity") for node in graph}  # find shortest distance
    predecessors = {node: None for node in graph}  # keep track of path
    distances[
        start
    ] = 0  # initialise distance of source node to 0 (since shortest path to source node is 0)
    priority_queue = [(0, start)]  # For efficient look up O(1) of next node to explore

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Stop condition for single source-target shortest path
        if current_node == target:
            break

        # Check if current distance is smaller in our set
        if current_distance > distances[current_node]:
            continue

        for neighbor in graph[current_node]:
            distance = current_distance + neighbor["distance"]
            if distance < distances[neighbor["node"]]:
                distances[neighbor["node"]] = distance
                predecessors[neighbor["node"]] = current_node
                heapq.heappush(priority_queue, (distance, neighbor["node"]))

    # Reconstruct the shortest path from the predecessor dictionary
    path = []
    while target:
        path.append(target)
        target = predecessors[target]
    path.reverse()  # Reverse to get path from start to target

    # Calculate the total energy cost
    total_energy_cost = 0
    for i in range(len(path) - 1):
        for neighbor in graph[path[i]]:
            if neighbor["node"] == path[i + 1]:
                total_energy_cost += neighbor["cost"]
                break

    return distances[path[-1]], path, total_energy_cost


def ucs_with_energy_constraint(graph, start, target, energy_budget):
    """
    Function to run UCS search algorithm with energy constraint (task 2)
    Parameters:
        graph: combined graph of the nodes, distances and costs
        start: start node
        target: target node
        energy_budget: maximum energy budget
    """
    visited = set()
    # Each item in the priority queue is (distance, energy_cost, current_node, path_so_far)
    priority_queue = [(0, 0, start, [])]

    while priority_queue:
        current_distance, current_energy_cost, current_node, path = heapq.heappop(
            priority_queue
        )

        if current_node in visited:
            continue

        new_path = path + [current_node]

        # If we reach the target node and satisfy the energy constraint, return the path, distance, and energy cost
        if current_node == target:
            return new_path, current_distance, current_energy_cost

        visited.add(current_node)

        for neighbor in graph[current_node]:
            if neighbor["node"] not in visited:
                # Sum the distance and energy cost to get to this neighbor
                new_distance = current_distance + neighbor["distance"]
                new_energy_cost = current_energy_cost + neighbor["cost"]

                # Only enqueue if the new energy cost is within the budget
                if new_energy_cost <= energy_budget:
                    heapq.heappush(
                        priority_queue,
                        (new_distance, new_energy_cost, neighbor["node"], new_path),
                    )

    # Return None if no path found within the energy constraint
    return None, None, None


def get_displacement(Coord, node_from, node_to):
    """
    Function to get the displacement between two nodes
    Parameters:
        node_from: node from which the displacement is calculated
        node_to: node to which the displacement is calculated
    """
    node_from_x, node_from_y = Coord[node_from]
    node_to_x, node_to_y = Coord[node_to]
    d = math.sqrt(
        pow(abs(node_from_x - node_to_x), 2) + pow(abs(node_from_y - node_to_y), 2)
    )
    return d


def a_star_search(graph, start, target, energy_budget, coord, h_factor=1):
    """
    Function to run A* search algorithm with energy constraint (task 3)
    Parameters:
        graph: combined graph of the nodes, distances and costs
        start: start node
        target: target node
        energy_budget: maximum energy budget
        coord: dictionary of coordinates of each node
        h_factor: factor to multiply heuristic by
    """
    # Initialize costs and visited set
    visited = set()
    g = {node: float("infinity") for node in graph}  # Distance from start to node
    h = {node: float("infinity") for node in graph}  # g + heuristic
    predecessors = {node: None for node in graph}  # Keep track of path
    energy_costs = {
        node: float("infinity") for node in graph
    }  # Energy cost from start to node

    g[start] = 0
    h[start] = get_displacement(coord, start, target)
    energy_costs[start] = 0

    # Each item in the priority queue is (estimated_cost_to_goal, cost_from_start, node, energy_cost)
    priority_queue = [(h[start], g[start], start, energy_costs[start])]

    while priority_queue:
        _, current_cost, current_node, current_energy = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == target:
            # Reconstruct path from start to target
            path = []
            while current_node:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.reverse()
            return path, current_cost, current_energy

        for neighbor in graph[current_node]:
            tentative_g = current_cost + neighbor["distance"]  # g(n)
            tentative_energy = current_energy + neighbor["cost"]  # energy cost

            if neighbor["node"] not in visited and tentative_energy <= energy_budget:
                if tentative_g < g[neighbor["node"]]:
                    g[neighbor["node"]] = tentative_g
                    energy_costs[neighbor["node"]] = tentative_energy
                    # f(n) = g(n) + h(n)
                    h[neighbor["node"]] = (
                        tentative_g
                        + get_displacement(coord, neighbor["node"], target) * h_factor
                    )
                    predecessors[neighbor["node"]] = current_node
                    heapq.heappush(
                        priority_queue,
                        (
                            h[neighbor["node"]],
                            g[neighbor["node"]],
                            neighbor["node"],
                            energy_costs[neighbor["node"]],
                        ),
                    )

    # No path found within energy constraint
    return None, None, None


def main():
    """
    Function to run main program
    """
    # Load json data
    Cost = load_json_file("Cost.json")
    G = load_json_file("G.json")
    Coord = load_json_file("Coord.json")
    Dist = load_json_file("Dist.json")

    # Combine the data
    combined_graph = combine_graphs(G, Dist, Cost)

    # Initialise constants
    start_node = "1"
    target_node = "50"
    energy_budget = 287932

    # Task 1
    print("=" * 5 + " Task 1 " + "=" * 5)
    shortest_distance, shortest_path, energy_cost = dijkstra(
        combined_graph, start_node, target_node
    )
    print("Shortest path:", "->".join(shortest_path) + ".")
    print(f"Shortest distance: {shortest_distance}.")
    print(f"Total energy cost: {energy_cost}.")

    # Task 2
    print("\n" + "=" * 5 + " Task 2 " + "=" * 5)
    shortest_path, shortest_distance, total_energy_cost = ucs_with_energy_constraint(
        combined_graph, start_node, target_node, energy_budget
    )
    if shortest_path:
        print("Shortest path:", "->".join(shortest_path) + ".")
        print(f"Shortest distance: {shortest_distance}.")
        print(f"Total energy cost: {total_energy_cost}.")
    else:
        print("No path found within the given energy constraint.")

    # Task 3
    print("\n" + "=" * 5 + " Task 3 " + "=" * 5)
    shortest_path, shortest_distance, total_energy_cost = a_star_search(
        combined_graph, start_node, target_node, energy_budget, Coord
    )
    if shortest_path:
        print("Shortest path:", "->".join(shortest_path) + ".")
        print(f"Shortest distance: {shortest_distance}.")
        print(f"Total energy cost: {total_energy_cost}.")
    else:
        print("No path found within the given energy constraint.")


if __name__ == "__main__":
    main()
