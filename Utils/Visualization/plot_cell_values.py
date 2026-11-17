import argparse

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch, Rectangle


def plot_cell_values(map_data, cell_values):
    width, height = map_data['map']['dimensions']
    values = np.asarray(cell_values)
    if values.shape != (height, width):
        raise ValueError(f'Expected cell_values shape {(height, width)}, got {values.shape}')

    figure, axis = plt.subplots(
        figsize=(max(5, width * 1.1), max(4, height * 1.1))
    )
    image = axis.imshow(
        values,
        cmap='YlGnBu',
        origin='lower',
        interpolation='nearest',
        vmin=1,
        vmax=10,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label('Cell value')

    axis.set_xticks(range(width))
    axis.set_yticks(range(height))
    axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=1.5)
    axis.tick_params(which='minor', bottom=False, left=False)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(-0.5, height - 0.5)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title('Environment cell values')

    fixed_obstacles = {tuple(cell) for cell in map_data['map'].get('obstacles', [])}
    movable_obstacles = map_data['map'].get('movable_obstacles', [])
    if isinstance(movable_obstacles, dict):
        movable_obstacles = [
            (path[0]['x'], path[0]['y'])
            for path in movable_obstacles.values()
            if path
        ]
    movable_obstacles = {tuple(cell) for cell in movable_obstacles}

    for x in range(width):
        for y in range(height):
            is_fixed = (x, y) in fixed_obstacles
            is_movable = (x, y) in movable_obstacles
            if is_fixed or is_movable:
                axis.add_patch(Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor='#20252b' if is_fixed else '#3f6670',
                    edgecolor='none', alpha=0.72,
                ))
            text_color = 'white' if is_fixed or is_movable else '#17242b'
            axis.text(x, y, str(values[y, x]), ha='center', va='center',
                      color=text_color, fontsize=12, fontweight='bold')

    for task in map_data.get('tasks', []):
        if 'start' in task:
            x, y = task['start']
            axis.text(x - 0.34, y - 0.34, 'P', ha='left', va='bottom',
                      color='#087f5b', fontsize=8, fontweight='bold')
        if 'goal' in task:
            x, y = task['goal']
            axis.text(x + 0.34, y + 0.34, 'D', ha='right', va='top',
                      color='#c2413b', fontsize=8, fontweight='bold')

    axis.legend(handles=[
        Patch(facecolor='#20252b', label='Fixed obstacle'),
        Patch(facecolor='#3f6670', label='Movable obstacle'),
    ], loc='upper left', bbox_to_anchor=(1.12, 1.0), frameon=False)
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', required=True, help='Map and task YAML file')
    parser.add_argument('-schedule', required=True, help='Simulation output YAML file')
    arguments = parser.parse_args()

    with open(arguments.map, encoding='utf-8') as map_file:
        map_data = yaml.safe_load(map_file)
    with open(arguments.schedule, encoding='utf-8') as schedule_file:
        output_data = yaml.safe_load(schedule_file)

    plot_cell_values(map_data, output_data['cell_values'])
