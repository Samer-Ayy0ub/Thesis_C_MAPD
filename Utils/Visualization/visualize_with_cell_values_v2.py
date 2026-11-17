import argparse
from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle


def format_cell_value(value):
    return f'{value:.4f}'.rstrip('0').rstrip('.')


def position_at_time(path, timestamp):
    if timestamp <= path[0]['t']:
        return path[0]['x'], path[0]['y']

    for previous, following in zip(path, path[1:]):
        if timestamp <= following['t']:
            duration = following['t'] - previous['t']
            fraction = (timestamp - previous['t']) / duration if duration else 0
            x = previous['x'] + (following['x'] - previous['x']) * fraction
            y = previous['y'] + (following['y'] - previous['y']) * fraction
            return x, y

    return path[-1]['x'], path[-1]['y']


def visualize(map_data, output_data, slow_factor=1):
    width, height = map_data['map']['dimensions']
    schedule = output_data['schedule']
    history = output_data.get('cell_values_history')
    if not history:
        history = [{'time': 0, 'values': output_data['cell_values']}]

    max_time = max((path[-1]['t'] for path in schedule.values() if path), default=0)
    frames_per_step = max(1, slow_factor)
    frame_count = (max_time + 1) * frames_per_step
    all_values = [
        value
        for snapshot in history
        for row in snapshot['values']
        for value in row
    ]
    value_min = min(all_values, default=0)
    value_max = max(all_values, default=1)
    if value_min == value_max:
        value_max = value_min + 1

    figure_height = 7
    figure_width = max(11, 14 * width / height + 3)
    figure, (path_axis, value_axis) = plt.subplots(
        1, 2, figsize=(figure_width, figure_height), constrained_layout=True
    )

    static_obstacles = {
        tuple(cell) for cell in map_data['map'].get('obstacles', [])
    }
    static_mask = np.zeros((height, width), dtype=bool)
    for x, y in static_obstacles:
        static_mask[y, x] = True

    for axis in (path_axis, value_axis):
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(-0.5, height - 0.5)
        axis.set_aspect('equal')
        axis.set_xticks(range(width))
        axis.set_yticks(range(height))
        axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
        axis.grid(which='minor', color='#aeb8bf', linewidth=0.8)
        axis.tick_params(which='minor', bottom=False, left=False)
        for x, y in static_obstacles:
            axis.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor='#d5d9dc', edgecolor='#aab2b8', zorder=2,
            ))

    for endpoint in map_data['map'].get('non_task_endpoints', []):
        for axis in (path_axis, value_axis):
            axis.add_patch(Circle(
                tuple(endpoint), 0.34, facecolor='#b9e2d2',
                edgecolor='#16835d', linewidth=1.2, zorder=3,
            ))

    initial_values = np.asarray(history[0]['values'])
    snapshot_times = [snapshot['time'] for snapshot in history]
    active_snapshot_index = 0
    value_cmap = plt.get_cmap('YlGnBu').copy()
    value_cmap.set_bad('#d5d9dc')
    value_image = value_axis.imshow(
        np.ma.array(initial_values, mask=static_mask),
        origin='lower',
        interpolation='nearest',
        cmap=value_cmap,
        vmin=value_min,
        vmax=value_max,
        zorder=0,
    )
    colorbar = figure.colorbar(
        value_image, ax=value_axis, fraction=0.046, pad=0.04
    )
    colorbar.set_label('Predicted cell value')

    value_texts = {}
    for x in range(width):
        for y in range(height):
            value_texts[(x, y)] = value_axis.text(
                x,
                y,
                format_cell_value(initial_values[y, x]),
                ha='center',
                va='center',
                color='#17242b',
                fontsize=9,
                fontweight='bold',
                zorder=4,
            )

    task_markers = []
    task_artists = []
    tasks = map_data.get('tasks', [])
    task_cmap = plt.get_cmap('turbo')
    for index, task in enumerate(tasks):
        color = task_cmap(index / max(1, len(tasks) - 1))
        offset_x = ((index % 3) - 1) * 0.13
        offset_y = ((index // 3) % 3 - 1) * 0.13
        pickup_x, pickup_y = task['start']
        delivery_x, delivery_y = task['goal']
        task_artists_for_task = []
        for axis, marker_size, offset_scale in (
            (path_axis, 70, 1.0),
            (value_axis, 38, 0.65),
        ):
            pickup = axis.scatter(
                [pickup_x + offset_x * offset_scale],
                [pickup_y + offset_y * offset_scale],
                marker='^',
                s=marker_size,
                color=[color],
                edgecolors='#17242b',
                linewidths=0.6,
                zorder=9,
            )
            delivery = axis.scatter(
                [delivery_x - offset_x * offset_scale],
                [delivery_y - offset_y * offset_scale],
                marker='s',
                s=marker_size,
                color=[color],
                edgecolors='#17242b',
                linewidths=0.6,
                zorder=9,
            )
            pickup.set_visible(False)
            delivery.set_visible(False)
            task_artists_for_task.extend((pickup, delivery))
        task_markers.append((task, *task_artists_for_task))
        task_artists.extend(task_artists_for_task)

    moving_entities = {
        name: path for name, path in schedule.items()
        if name.startswith('obs_') and path
    }
    agent_entities = {
        name: path for name, path in schedule.items()
        if not name.startswith('obs_') and path
    }
    path_markers = {}
    path_labels = {}
    value_markers = {}

    for name, path in {**agent_entities, **moving_entities}.items():
        is_obstacle = name.startswith('obs_')
        color = '#66747d' if is_obstacle else '#176b87'
        radius = 0.28 if is_obstacle else 0.31
        x, y = position_at_time(path, 0)
        path_markers[name] = Circle(
            (x, y), radius, facecolor=color,
            edgecolor='white', linewidth=1, zorder=6,
        )
        path_axis.add_patch(path_markers[name])
        if not is_obstacle:
            label = name.replace('agent', 'A')
            path_labels[name] = path_axis.text(
                x, y, label, ha='center', va='center',
                color='white', fontsize=8, zorder=7,
            )
        if is_obstacle:
            value_markers[name] = Rectangle(
                (x - 0.43, y - 0.43), 0.86, 0.86,
                facecolor='none', edgecolor='#e36b35',
                linewidth=2, zorder=5,
            )
            value_axis.add_patch(value_markers[name])

    path_axis.set_title('Simulation')
    value_axis.set_title('Cell values')
    path_axis.set_xlabel('X')
    value_axis.set_xlabel('X')
    path_axis.set_ylabel('Y')
    value_axis.set_ylabel('Y')
    path_axis.legend(handles=[
        Patch(facecolor='#d5d9dc', edgecolor='#aab2b8', label='Fixed wall'),
        Patch(facecolor='#66747d', label='Movable obstacle'),
        Patch(facecolor='#176b87', label='Agent'),
        Line2D([], [], marker='^', linestyle='none',
               color='#56636a', label='Pickup'),
        Line2D([], [], marker='s', linestyle='none',
               color='#56636a', label='Delivery'),
    ], loc='upper left', bbox_to_anchor=(0, 1.14), ncol=3, frameon=False)
    time_label = path_axis.text(
        0.98, 0.98, 'Time: 0.0',
        transform=path_axis.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        fontweight='bold',
        color='#17242b',
        bbox={'facecolor': 'white', 'edgecolor': '#66747d', 'alpha': 0.9},
        zorder=10,
    )

    def update(frame):
        nonlocal active_snapshot_index
        timestamp = frame / frames_per_step
        time_label.set_text(f'Time: {timestamp:.1f}')
        completed_times = output_data.get('completed_tasks_times', {})
        for task, pickup, delivery, value_pickup, value_delivery in task_markers:
            start_time = task.get('start_time', 0)
            end_time = completed_times.get(task['task_name'], max_time)
            visible = start_time <= timestamp <= end_time
            pickup.set_visible(visible)
            delivery.set_visible(visible)
            value_pickup.set_visible(visible)
            value_delivery.set_visible(visible)

        for name, path in {**agent_entities, **moving_entities}.items():
            position = position_at_time(path, timestamp)
            path_markers[name].center = position
            if name in path_labels:
                path_labels[name].set_position(position)
            if name in value_markers:
                x, y = position
                value_markers[name].set_xy((x - 0.43, y - 0.43))

        snapshot_index = max(
            0, bisect_right(snapshot_times, timestamp) - 1
        )
        if snapshot_index != active_snapshot_index:
            active_snapshot_index = snapshot_index
            values = np.asarray(history[snapshot_index]['values'])
            value_image.set_data(np.ma.array(values, mask=static_mask))
            for (x, y), text in value_texts.items():
                value = values[y, x]
                text.set_text(format_cell_value(value))
                if (x, y) in static_obstacles:
                    text.set_color('#17242b')
                else:
                    text.set_color(
                        'white'
                        if (value - value_min) / (value_max - value_min) > 0.58
                        else '#17242b'
                    )

        return [
            time_label,
            value_image,
            *path_markers.values(),
            *path_labels.values(),
            *value_markers.values(),
            *value_texts.values(),
            *task_artists,
        ]

    animation = FuncAnimation(
        figure, update, frames=frame_count, interval=10, blit=True, repeat=False
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', required=True, help='Map and task YAML file')
    parser.add_argument('-schedule', required=True, help='Demo output YAML file')
    parser.add_argument('-slow_factor', default=1, type=int)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as map_file:
        map_yaml = yaml.safe_load(map_file)
    with open(args.schedule, encoding='utf-8') as schedule_file:
        schedule_yaml = yaml.safe_load(schedule_file)

    visualize(map_yaml, schedule_yaml, slow_factor=args.slow_factor)
