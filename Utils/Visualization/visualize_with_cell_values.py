#!/usr/bin/env python3
import argparse
from bisect import bisect_right
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import RoothPath
from Utils.Visualization.visualize import Animation as BaseAnimation, Colors


def format_cell_value(value):
    return f'{value:.4f}'.rstrip('0').rstrip('.')


class Animation(BaseAnimation):
    def __init__(self, map, schedule, slow_factor=10):
        history = schedule.get('cell_values_history')
        if not history:
            history = [{'time': 0, 'values': schedule['cell_values']}]

        width, height = map['map']['dimensions']
        expected_shape = (height, width)
        self.cell_values_history = []
        for snapshot in history:
            values = np.asarray(snapshot['values'])
            if values.shape != expected_shape:
                raise ValueError(
                    f"Expected cell values shape {expected_shape}, got {values.shape}"
                )
            self.cell_values_history.append((snapshot['time'], values))
        self.snapshot_times = [
            time for time, _ in self.cell_values_history
        ]

        all_values = np.concatenate([
            values.ravel() for _, values in self.cell_values_history
        ])
        self.value_min = float(np.min(all_values))
        self.value_max = float(np.max(all_values))
        if self.value_min == self.value_max:
            self.value_max = self.value_min + 1
        self.active_snapshot_index = 0

        super().__init__(map, schedule, slow_factor)

        original_width = 4 * width / height
        panel_gap = 0.25
        colorbar_width = 0.5
        figure_width = 2 * original_width + panel_gap + colorbar_width
        self.fig.set_size_inches(figure_width, 4, forward=True)
        self.ax.set_position((0, 0, original_width / figure_width, 1))

        value_axis_left = (original_width + panel_gap) / figure_width
        self.value_ax = self.fig.add_axes(
            (value_axis_left, 0, original_width / figure_width, 1)
        )
        self.value_colorbar_ax = self.fig.add_axes(
            (
                (2 * original_width + panel_gap + 0.12) / figure_width,
                0.12,
                0.08 / figure_width,
                0.76,
            )
        )
        self._create_cell_value_grid(width, height)

    def _create_cell_value_grid(self, width, height):
        initial_values = self.cell_values_history[0][1]
        self.value_image = self.value_ax.imshow(
            initial_values,
            origin='lower',
            interpolation='nearest',
            cmap='YlGnBu',
            vmin=self.value_min,
            vmax=self.value_max,
            zorder=0,
        )
        colorbar = self.fig.colorbar(
            self.value_image, cax=self.value_colorbar_ax
        )
        colorbar.set_label('Cell value')

        self.value_ax.add_patch(Rectangle(
            (-0.5, -0.5), width, height,
            facecolor='none', edgecolor='red',
        ))
        for x in range(width):
            for y in range(height):
                self.value_ax.add_patch(Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor='none', edgecolor='black',
                ))

        fixed_obstacles = {
            tuple(cell) for cell in self.map['map'].get('obstacles', [])
        }
        for x, y in fixed_obstacles:
            self.value_ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor='#444444', edgecolor='black',
            ))

        for endpoint in self.map['map'].get('non_task_endpoints', []):
            self.value_ax.add_patch(Circle(
                tuple(endpoint), 0.4, facecolor='green', edgecolor='black',
            ))

        self.value_task_patches = {}
        for task in self.map['tasks']:
            task_color = self.tasks[task['task_name']][0].get_facecolor()
            pickup_x, pickup_y = task['start']
            delivery_x, delivery_y = task['goal']
            pickup = Rectangle(
                (pickup_x - 0.25, pickup_y - 0.25), 0.5, 0.5,
                facecolor=task_color, edgecolor='black', alpha=0,
            )
            delivery = RegularPolygon(
                (delivery_x, delivery_y - 0.05), 3, radius=0.2,
                facecolor=task_color, edgecolor='black', alpha=0,
            )
            self.value_ax.add_patch(pickup)
            self.value_ax.add_patch(delivery)
            self.value_task_patches[task['task_name']] = (pickup, delivery)

        for agent in self.map['agents']:
            if 'goal' in agent:
                goal_x, goal_y = agent['goal']
                self.value_ax.add_patch(Rectangle(
                    (goal_x - 0.25, goal_y - 0.25), 0.5, 0.5,
                    facecolor=Colors[0],
                    edgecolor='black', alpha=0.5,
                ))

        self.value_ax.set_xlim(-0.5, width - 0.5)
        self.value_ax.set_ylim(-0.5, height - 0.5)
        self.value_ax.set_aspect('equal')
        self.value_time_label = self.value_ax.set_title('Cell values | Time: 0')
        self.fixed_obstacles = fixed_obstacles
        self.value_labels_image = self.value_ax.imshow(
            self._make_value_labels(initial_values),
            origin='lower',
            interpolation='nearest',
            extent=(-0.5, width - 0.5, -0.5, height - 0.5),
            zorder=5,
        )

    def _value_text_color(self, value, is_fixed_obstacle=False):
        if is_fixed_obstacle:
            return 'white'
        normalized_value = (value - self.value_min) / (
            self.value_max - self.value_min
        )
        return 'white' if normalized_value > 0.58 else 'black'

    def _make_value_labels(self, values):
        height, width = values.shape
        pixels_per_cell = 40
        strings = [
            [format_cell_value(value) for value in row]
            for row in values
        ]
        font_path = font_manager.findfont('DejaVu Sans')
        font_size = max(6, int(pixels_per_cell * 0.55))
        font = ImageFont.truetype(font_path, font_size)
        max_text_width = pixels_per_cell * 0.9
        while max(
            font.getlength(text)
            for row in strings
            for text in row
        ) > max_text_width and font_size > 6:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)

        image = Image.new(
            'RGBA', (width * pixels_per_cell, height * pixels_per_cell)
        )
        draw = ImageDraw.Draw(image)
        for y, row in enumerate(strings):
            for x, text in enumerate(row):
                fill = self._value_text_color(
                    values[y, x], (x, y) in self.fixed_obstacles
                )
                center = (
                    x * pixels_per_cell + pixels_per_cell / 2,
                    (height - y - 1) * pixels_per_cell + pixels_per_cell / 2,
                )
                draw.text(center, text, font=font, fill=fill, anchor='mm')
        return np.flipud(np.asarray(image))

    def init_func(self):
        super().init_func()
        if not hasattr(self, 'value_labels_image'):
            return self._dynamic_artists()
        return self._dynamic_artists() + [
            self.value_image,
            self.value_labels_image,
            self.value_time_label,
        ]

    def _dynamic_artists(self):
        return [
            *self.agents.values(),
            *self.artists,
            *(patch for task_patches in self.tasks.values() for patch in task_patches),
            *(
                patch
                for task_patches in getattr(self, 'value_task_patches', {}).values()
                for patch in task_patches
            ),
        ]

    def animate_func(self, frame):
        super().animate_func(frame)
        timestamp = frame / self.slow_factor
        self.value_time_label.set_text(f'Cell values | Time: {timestamp:.1f}')

        for task_name, (pickup, delivery) in self.value_task_patches.items():
            pickup.set_alpha(self.tasks[task_name][0].get_alpha())
            delivery.set_alpha(self.tasks[task_name][1].get_alpha())

        snapshot_index = max(0, bisect_right(self.snapshot_times, timestamp) - 1)
        if snapshot_index != self.active_snapshot_index:
            self.active_snapshot_index = snapshot_index
            values = self.cell_values_history[snapshot_index][1]
            self.value_image.set_data(values)
            self.value_labels_image.set_data(self._make_value_labels(values))

        return self._dynamic_artists() + [
            self.value_time_label,
            self.value_image,
            self.value_labels_image,
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', help='input file containing map')
    parser.add_argument('-schedule', help='schedule for agents')
    parser.add_argument(
        '-slow_factor', help='Slow factor of visualization', default=1, type=int
    )
    parser.add_argument(
        '--video',
        dest='video',
        default=None,
        help='output video file (or leave empty to show on screen)',
    )
    parser.add_argument('--speed', type=int, default=1, help='speedup-factor')
    args = parser.parse_args()

    if args.map is None:
        with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
            config = json.load(json_file)
        args.map = os.path.join(
            RoothPath.get_root(),
            os.path.join(
                config['input_path'],
                config['input_name'] + config['visual_postfix'],
            ),
        )
        args.schedule = os.path.join(RoothPath.get_root(), 'output.yaml')

    with open(args.map) as map_file:
        map_data = yaml.load(map_file, Loader=yaml.FullLoader)

    with open(args.schedule) as schedule_file:
        schedule_data = yaml.load(schedule_file, Loader=yaml.FullLoader)

    animation = Animation(map_data, schedule_data, slow_factor=args.slow_factor)

    if args.video:
        animation.save(args.video, args.speed)
    else:
        animation.show()
