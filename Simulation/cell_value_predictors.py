"""Predictors that map observed cell visits to cell values for obstacle scoring."""

from collections import deque


def frequency_predictor(cell_visit_counts, context):
    """Use observed visit frequency as the predicted cell value."""
    return dict(cell_visit_counts)


def manhattan_task_predictor(cell_visit_counts, context):
    """Count unfinished-task Manhattan routes, moving along x before y."""
    predicted_values = {cell: 0 for cell in cell_visit_counts}

    for task in context['unfinished_tasks']:
        x, y = task['start']
        goal_x, goal_y = task['goal']
        predicted_values[(x, y)] += 1

        while x != goal_x:
            x += 1 if goal_x > x else -1
            predicted_values[(x, y)] += 1

        while y != goal_y:
            y += 1 if goal_y > y else -1
            predicted_values[(x, y)] += 1

    return predicted_values


def fixed_obstacle_shortest_path_predictor(cell_visit_counts, context):
    """Estimate shortest-path visit probability while avoiding fixed obstacles."""
    dimensions = context['dimensions']
    obstacles = context['obstacles']
    predicted_values = {cell: 0 for cell in cell_visit_counts}

    def shortest_path_counts(origin):
        distances = {origin: 0}
        path_counts = {origin: 1}
        queue = deque([origin])

        while queue:
            x, y = queue.popleft()
            next_distance = distances[(x, y)] + 1
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = neighbor
                if not (0 <= nx < dimensions[0] and 0 <= ny < dimensions[1]):
                    continue
                if neighbor in obstacles:
                    continue

                if neighbor not in distances:
                    distances[neighbor] = next_distance
                    path_counts[neighbor] = path_counts[(x, y)]
                    queue.append(neighbor)
                elif distances[neighbor] == next_distance:
                    path_counts[neighbor] += path_counts[(x, y)]

        return distances, path_counts

    for task in context['unfinished_tasks']:
        start = tuple(task['start'])
        goal = tuple(task['goal'])
        if start not in cell_visit_counts or goal not in cell_visit_counts:
            continue
        if start in obstacles or goal in obstacles:
            continue

        start_distances, start_path_counts = shortest_path_counts(start)
        if goal not in start_distances:
            continue
        goal_distances, goal_path_counts = shortest_path_counts(goal)
        shortest_distance = start_distances[goal]
        total_shortest_paths = start_path_counts[goal]

        for cell, distance_from_start in start_distances.items():
            distance_to_goal = goal_distances.get(cell)
            if distance_to_goal is not None and distance_from_start + distance_to_goal == shortest_distance:
                paths_through_cell = start_path_counts[cell] * goal_path_counts[cell]
                predicted_values[cell] += paths_through_cell / total_shortest_paths

    return predicted_values


def detour_cost_predictor(cell_visit_counts, context):
    """Score cells by the extra task distance caused by blocking them."""
    dimensions = context['dimensions']
    obstacles = context['obstacles']
    free_cells = [
        cell for cell in cell_visit_counts
        if cell not in obstacles
    ]
    predicted_values = {cell: 0 for cell in cell_visit_counts}
    unreachable_penalty = len(free_cells) + 1

    def shortest_distances(origin, blocked_cell=None):
        if origin in obstacles or origin == blocked_cell:
            return {}

        distances = {origin: 0}
        queue = deque([origin])
        while queue:
            x, y = queue.popleft()
            next_distance = distances[(x, y)] + 1
            for neighbor in (
                (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
            ):
                nx, ny = neighbor
                if not (0 <= nx < dimensions[0] and 0 <= ny < dimensions[1]):
                    continue
                if neighbor in obstacles or neighbor == blocked_cell:
                    continue
                if neighbor not in distances:
                    distances[neighbor] = next_distance
                    queue.append(neighbor)
        return distances

    for task in context['unfinished_tasks']:
        start = tuple(task['start'])
        goal = tuple(task['goal'])
        if start not in cell_visit_counts or goal not in cell_visit_counts:
            continue

        start_distances = shortest_distances(start)
        baseline_distance = start_distances.get(goal)
        if baseline_distance is None:
            continue
        goal_distances = shortest_distances(goal)

        for cell in free_cells:
            distance_from_start = start_distances.get(cell)
            distance_to_goal = goal_distances.get(cell)
            if (
                distance_from_start is None
                or distance_to_goal is None
                or distance_from_start + distance_to_goal != baseline_distance
            ):
                continue

            blocked_distance = shortest_distances(start, blocked_cell=cell).get(goal)
            if blocked_distance is None:
                predicted_values[cell] += unreachable_penalty
            else:
                predicted_values[cell] += blocked_distance - baseline_distance

    return predicted_values


def null_predictor(cell_visit_counts, context):
    """Return a constant value for all cells."""
    return {cell: 1 for cell in cell_visit_counts}