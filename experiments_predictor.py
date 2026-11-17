import copy
import io
import json
import multiprocessing
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
from matplotlib.lines import Line2D

import RoothPath
from Simulation import cell_value_predictors
from Simulation.TP_with_recovery import TokenPassingRecovery
from Simulation.simulation_new_recovery import SimulationNewRecovery
from Simulation.tasks_and_delays_maker import gen_tasks_and_delays
from Utils.pool_with_subprocess import PoolWithSubprocess


# Select exactly one predictor for this experiment run.
CELL_VALUE_PREDICTOR = cell_value_predictors.null_predictor
GAMMA = 0
NUM_RUNS = 10
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

SCENARIOS = {
    "Scenario_1_The_Sliding_Puzzle": {"alpha": 1.0, "radius": 3, "laziness": 1},
    "Scenario_2_Proactive_Clearance": {"alpha": 1.0, "radius": 5, "laziness": 4},
    "Scenario_3_Heavy_Duty_Bottleneck": {"alpha": 20.0, "radius": 5, "laziness": 3},
    "Scenario_4_Cheap_Deep_Relocation": {"alpha": 0.1, "radius": 3, "laziness": 5},
}

SWEEP_VALUES = {
    "Alpha": [0.1, 0.5, 1.0],
    "Radius": [3, 5, 7],
    "Laziness": [1, 2, 5],
}

COLORS = {
    "Alpha": {
        "delivery_time": "#d62728",
        "makespan": "#ff7f0e",
        "sim_time": "#2ca02c",
        "terraforming_cost": "#9467bd",
    },
    "Radius": {
        "delivery_time": "#1f77b4",
        "makespan": "#9467bd",
        "sim_time": "#d62728",
        "terraforming_cost": "#8c564b",
    },
    "Laziness": {
        "delivery_time": "#2ca02c",
        "makespan": "#8c564b",
        "sim_time": "#1f77b4",
        "terraforming_cost": "#e377c2",
    },
}


class LogCatcher:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        return self

    def __exit__(self, *_exc_info):
        sys.stdout = self._original_stdout
        return None if not _exc_info else False

    def get_critical_warnings(self):
        warnings = set()
        for line in self.captured_output.getvalue().split('\n'):
            line = line.strip()
            if 'fatal exception' in line.lower() or 'fatal error' in line.lower():
                warnings.add(line)
        return list(warnings)


def verify_no_collisions(simulation, tp):
    states = defaultdict(int)
    paths = dict(simulation.actual_paths)
    max_time = simulation.get_time()

    for obstacle_id, path in tp.get_token()['obstacles_paths'].items():
        path_by_time = {step['t']: step for step in path}
        last_pos = path[0] if path else {'x': 0, 'y': 0}
        continuous_path = []
        for time in range(max_time + 1):
            if time in path_by_time:
                last_pos = path_by_time[time]
            continuous_path.append({
                't': time,
                'x': last_pos['x'],
                'y': last_pos['y'],
            })
        paths[obstacle_id] = continuous_path

    for path in paths.values():
        for step in path:
            state = (step['t'], step['x'], step['y'])
            states[state] += 1
            if states[state] > 1:
                return True
    return False


def generate_map_preview(yaml_path, output_filename):
    with open(yaml_path, 'r') as map_file:
        map_data = yaml.safe_load(map_file)

    map_info = map_data['map']
    dimensions = map_info['dimensions']
    static_obstacles = map_info.get('obstacles', [])
    movable_obstacles = map_info.get('movable_obstacles', [])
    parking_endpoints = map_info.get('non_task_endpoints', [])
    task_endpoints = {
        tuple(endpoint)
        for endpoint in (
            map_info.get('start_locations', []) + map_info.get('goal_locations', [])
        )
    }

    _, ax = plt.subplots(figsize=(dimensions[0], dimensions[1]))
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])
    ax.set_xticks(range(dimensions[0] + 1))
    ax.set_yticks(range(dimensions[1] + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_aspect('equal')

    def draw_square(coords, color, edgecolor='black', hatch=None, zorder=1,
                    linestyle='solid', linewidth=1):
        ax.add_patch(patches.Rectangle(
            (coords[0], coords[1]), 1, 1, facecolor=color,
            edgecolor=edgecolor, hatch=hatch, zorder=zorder,
            linestyle=linestyle, linewidth=linewidth,
        ))

    for endpoint in task_endpoints:
        draw_square(
            endpoint, '#e5f5e0', edgecolor='#31a354', zorder=0.5,
            linestyle='--', linewidth=1.5,
        )
    for endpoint in parking_endpoints:
        draw_square(endpoint, '#cce5ff', edgecolor='#0066cc')
        ax.text(
            endpoint[0] + 0.5, endpoint[1] + 0.5, 'EP',
            color='#0066cc', ha='center', va='center',
            fontsize=10, fontweight='bold',
        )
    for obstacle in static_obstacles:
        draw_square(obstacle, '#333333', zorder=2)
    for obstacle in movable_obstacles:
        draw_square(obstacle, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    for agent in map_data.get('agents', []):
        x, y = agent['start']
        center = (x + 0.5, y + 0.5)
        ax.add_patch(patches.Circle(
            center, 0.35, facecolor='#0033cc', edgecolor='black', zorder=4,
        ))
        ax.text(
            center[0], center[1], agent['name'].replace('agent', 'A'),
            color='white', ha='center', va='center', fontsize=9,
            fontweight='bold', zorder=5,
        )

    legend = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(
            facecolor='#ff9900', edgecolor='#cc0000', hatch='///',
            label='Movable Obstacle',
        ),
        patches.Patch(
            facecolor='#cce5ff', edgecolor='#0066cc', label='Parking Endpoint',
        ),
        patches.Patch(
            facecolor='#e5f5e0', edgecolor='#31a354', linestyle='--',
            linewidth=1.5, label='Task Endpoint',
        ),
        Line2D(
            [0], [0], marker='o', color='w', label='Agent',
            markerfacecolor='#0033cc', markersize=12, markeredgecolor='black',
        ),
    ]
    ax.legend(
        handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.05),
        ncol=3, frameon=False,
    )
    ax.set_title('Map Preview', fontsize=16, pad=15)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')


def load_experiment_inputs(yaml_path):
    with open(yaml_path, 'r') as map_file:
        param = yaml.safe_load(map_file)

    map_info = param['map']
    return (
        param,
        param['agents'],
        map_info['dimensions'],
        map_info.get('obstacles', []),
        map_info.get('movable_obstacles', []),
        map_info['non_task_endpoints'],
    )


def make_tasks_and_delays(param, agents, seed):
    random.seed(seed)
    return gen_tasks_and_delays(
        agents,
        param['map']['start_locations'],
        param['map']['goal_locations'],
        param.get('n_tasks', 20),
        param.get('task_freq', 1.0),
        param.get('n_delays_per_agent', 0),
    )


def advance_simulation_and_record_delivery_times(
    simulation, tp, task_locations, delivery_start_times, delivery_goal_times,
):
    previous_assignments = {
        agent_name: task_info['task_name']
        for agent_name, task_info in tp.get_token()['agents_to_tasks'].items()
    }
    previous_positions = {
        agent_name: (path[-1]['x'], path[-1]['y'])
        for agent_name, path in simulation.actual_paths.items()
    }
    previous_path_lengths = {
        agent_name: len(path)
        for agent_name, path in simulation.actual_paths.items()
    }

    simulation.time_forward(tp)
    current_time = simulation.get_time()

    for agent_name, task_info in tp.get_token()['agents_to_tasks'].items():
        task_name = task_info['task_name']
        if task_name == 'safe_idle':
            continue

        task_start, task_goal = task_locations[task_name]
        path = simulation.actual_paths[agent_name]
        newly_assigned = previous_assignments.get(agent_name) != task_name

        # The task may be assigned while the agent is already at its start.
        if task_name not in delivery_start_times and (
            newly_assigned and previous_positions[agent_name] == task_start
        ):
            delivery_start_times[task_name] = current_time

        for step in path[previous_path_lengths[agent_name]:]:
            position = (step['x'], step['y'])
            if task_name not in delivery_start_times and position == task_start:
                delivery_start_times[task_name] = step['t']
            if (
                task_name in delivery_start_times
                and task_name not in delivery_goal_times
                and position == task_goal
            ):
                delivery_goal_times[task_name] = step['t']


def calculate_average_delivery_time(
    completed_task_names, delivery_start_times, delivery_goal_times,
):
    missing_times = [
        task_name
        for task_name in completed_task_names
        if (
            task_name not in delivery_start_times
            or task_name not in delivery_goal_times
        )
    ]
    if missing_times:
        raise ValueError(
            f'Missing delivery arrival times for tasks: {sorted(missing_times)}'
        )

    delivery_times = [
        delivery_goal_times[task_name] - delivery_start_times[task_name]
        for task_name in completed_task_names
    ]
    if any(delivery_time < 0 for delivery_time in delivery_times):
        raise ValueError('A task goal was reached before its start location.')
    return sum(delivery_times) / len(delivery_times) if delivery_times else 0


def run_baseline(tasks, delays, agents, dimensions, obstacles, endpoints):
    simulation = SimulationNewRecovery(tasks, agents, delays=delays)
    tp = TokenPassingRecovery(
        agents, dimensions, obstacles, endpoints, simulation,
        a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
        terraforming_radius=1, new_recovery=True, movable_obstacles=[],
        alpha=1.0, laziness=1, cell_value_predictor=CELL_VALUE_PREDICTOR,
    )
    task_locations = {
        task['task_name']: (tuple(task['start']), tuple(task['goal']))
        for task in tasks
    }
    delivery_start_times = {}
    delivery_goal_times = {}
    while (
        tp.get_completed_tasks() != len(tasks)
        and simulation.get_time() < MAX_SIM_TIME
    ):
        advance_simulation_and_record_delivery_times(
            simulation, tp, task_locations,
            delivery_start_times, delivery_goal_times,
        )

    if (
        simulation.get_time() >= MAX_SIM_TIME
        or verify_no_collisions(simulation, tp)
    ):
        return None

    completed_times = tp.get_completed_tasks_times()
    makespan = max(completed_times.values()) if completed_times else 0
    average_delivery_time = calculate_average_delivery_time(
        completed_times, delivery_start_times, delivery_goal_times,
    )
    return makespan, average_delivery_time


def baseline_worker_task(payload):
    param, agents, dimensions, obstacles, movable_obstacles, endpoints = (
        load_experiment_inputs(payload['yaml_path'])
    )
    tasks, delays = make_tasks_and_delays(param, agents, payload['seed'])

    results = {}
    for name, static_obstacles in (
        ('LB', obstacles),
        ('UB', obstacles + movable_obstacles),
    ):
        with LogCatcher():
            try:
                outcome = run_baseline(
                    copy.deepcopy(tasks), copy.deepcopy(delays),
                    agents, dimensions, static_obstacles, endpoints,
                )
            except SystemExit:
                outcome = None
        results[name] = outcome

    return {
        'seed': payload['seed'],
        'lb_failed': results['LB'] is None,
        'lb_makespan': results['LB'][0] if results['LB'] else 0,
        'lb_delivery_time': results['LB'][1] if results['LB'] else 0,
        'ub_failed': results['UB'] is None,
        'ub_makespan': results['UB'][0] if results['UB'] else 0,
        'ub_delivery_time': results['UB'][1] if results['UB'] else 0,
    }


def worker_task(payload):
    param, agents, dimensions, obstacles, movable_obstacles, endpoints = (
        load_experiment_inputs(payload['yaml_path'])
    )
    tasks, delays = make_tasks_and_delays(param, agents, payload['seed'])

    with LogCatcher() as catcher:
        try:
            simulation = SimulationNewRecovery(tasks, agents, delays=delays)
            tp = TokenPassingRecovery(
                agents, dimensions, obstacles, endpoints, simulation,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=payload['radius'], new_recovery=True,
                movable_obstacles=movable_obstacles,
                alpha=payload['alpha'], laziness=payload['laziness'],
                gamma=payload['gamma'],
                cell_value_predictor=CELL_VALUE_PREDICTOR,
            )
            task_locations = {
                task['task_name']: (tuple(task['start']), tuple(task['goal']))
                for task in tasks
            }
            delivery_start_times = {}
            delivery_goal_times = {}
            while (
                tp.get_completed_tasks() != len(tasks)
                and simulation.get_time() < MAX_SIM_TIME
            ):
                advance_simulation_and_record_delivery_times(
                    simulation, tp, task_locations,
                    delivery_start_times, delivery_goal_times,
                )
            warnings = catcher.get_critical_warnings()
        except SystemExit:
            return {
                'failed': True,
                'seed': payload['seed'],
                'warnings': catcher.get_critical_warnings(),
            }

        if simulation.get_time() >= MAX_SIM_TIME:
            warnings.append('TIMEOUT: MAX_SIM_TIME reached.')
            return {'failed': True, 'seed': payload['seed'], 'warnings': warnings}

        if verify_no_collisions(simulation, tp):
            warnings.append('PHYSICAL COLLISION DETECTED DURING VALIDATION!')
            return {'failed': True, 'seed': payload['seed'], 'warnings': warnings}

        completed_times = tp.get_completed_tasks_times()
        makespan = max(completed_times.values()) if completed_times else 0
        average_delivery_time = calculate_average_delivery_time(
            completed_times, delivery_start_times, delivery_goal_times,
        )

        obstacle_moves = sum(
            1
            for path in tp.get_token()['obstacles_paths'].values()
            for previous, current in zip(path, path[1:])
            if (previous['x'], previous['y']) != (current['x'], current['y'])
        )
        terraforming_cost = payload['alpha'] * obstacle_moves

        return {
            'makespan': makespan,
            'delivery_time': average_delivery_time,
            'sim_time': simulation.get_algo_time(),
            'terraforming_cost': terraforming_cost,
            'seed': payload['seed'],
            'failed': False,
            'warnings': warnings,
        }


def save_metric_plot(
    output_path, scenario_name, parameter_name, values, averages, deviations,
    metric_name, ylabel, marker, color, fixed_text, baselines=None,
):
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        values, averages, yerr=deviations, fmt=f'-{marker}', color=color,
        ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5,
        markersize=8, label=f'Avg {metric_name}',
    )
    if baselines:
        for key, label, baseline_color in (
            ('LB', 'Lower Bound (No Movable Obs)', 'green'),
            ('UB', 'Upper Bound (Static M.O.)', 'red'),
        ):
            baseline = baselines[key].get(metric_name)
            if baseline is not None:
                plt.axhline(
                    baseline, color=baseline_color, linestyle='--',
                    linewidth=2, label=label,
                )

    plt.title(
        f'{scenario_name}: {metric_name.replace("_", " ").title()} '
        f'vs {parameter_name}',
        fontsize=14, fontweight='bold', pad=15,
    )
    plt.xlabel(f'{parameter_name} Value', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.text(
        0.05, 0.95, fixed_text, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(
            boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray',
        ),
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    if baselines:
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def collect_baselines(results):
    lower_bound = [
        result for result in results if not result['lb_failed']
    ]
    upper_bound = [
        result for result in results if not result['ub_failed']
    ]
    return {
        'LB': {
            'makespan': (
                float(np.mean([result['lb_makespan'] for result in lower_bound]))
                if lower_bound else None
            ),
            'delivery_time': (
                float(np.mean([result['lb_delivery_time'] for result in lower_bound]))
                if lower_bound else None
            ),
        },
        'UB': {
            'makespan': (
                float(np.mean([result['ub_makespan'] for result in upper_bound]))
                if upper_bound else None
            ),
            'delivery_time': (
                float(np.mean([result['ub_delivery_time'] for result in upper_bound]))
                if upper_bound else None
            ),
        },
    }


def main():
    predictor_name = CELL_VALUE_PREDICTOR.__name__
    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    map_name = config['input_name']
    map_name_clean = map_name.replace('.yaml', '')
    map_path = os.path.join(
        RoothPath.get_root(), config['input_path'], map_name,
    )
    output_dir = os.path.join(
        RoothPath.get_root(), 'predictors plots', predictor_name, map_name_clean,
    )
    os.makedirs(output_dir, exist_ok=True)

    preview_path = os.path.join(output_dir, 'map_preview.png')
    generate_map_preview(map_path, preview_path)
    print(f'Predictor: {predictor_name}')
    print(f'Gamma: {GAMMA}')
    print(f'Test Map: {map_name}')
    print(f'Output Directory: {output_dir}')
    print(f'Map Preview Generated: {preview_path}')

    cores = max(1, multiprocessing.cpu_count() - 2)
    baseline_payloads = [
        {'yaml_path': map_path, 'seed': seed}
        for seed in range(1, NUM_RUNS + 1)
    ]
    print('>>> Calculating Standard Baselines (Lower and Upper Bounds) <<<')
    pool = PoolWithSubprocess(processes=cores)
    try:
        baseline_results = pool.map_async(
            baseline_worker_task, baseline_payloads,
        ).get(timeout=86400)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit(1)

    baselines = collect_baselines(baseline_results)
    print(f'Baselines: {baselines}')
    all_results = {
        'predictor': predictor_name,
        'gamma': GAMMA,
        'num_runs': NUM_RUNS,
        'baselines': baselines,
        'scenarios': {},
    }

    for scenario_name, base_params in SCENARIOS.items():
        print(f'>>> Executing Scenario: {scenario_name} <<<')
        scenario_dir = os.path.join(output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        scenario_results = {'baselines': baselines, 'gamma': GAMMA}

        for parameter_name, sweep_values in SWEEP_VALUES.items():
            metric_results = {
                'values': sweep_values,
                'base_settings': base_params.copy(),
                'avg_delivery_time': [],
                'std_delivery_time': [],
                'avg_makespan': [],
                'std_makespan': [],
                'avg_sim_time': [],
                'std_sim_time': [],
                'avg_terraforming_cost': [],
                'std_terraforming_cost': [],
                'raw_data': {},
            }

            for value in sweep_values:
                print(
                    f'  {parameter_name} = {value} | '
                    f'Parallel Runs: {NUM_RUNS}...'
                )
                payloads = []
                for seed in range(1, NUM_RUNS + 1):
                    params = base_params.copy()
                    params[parameter_name.lower()] = value
                    payloads.append({
                        'yaml_path': map_path,
                        'seed': seed,
                        'gamma': GAMMA,
                        'alpha': params['alpha'],
                        'radius': params['radius'],
                        'laziness': params['laziness'],
                    })

                pool = PoolWithSubprocess(processes=cores)
                try:
                    results = pool.map_async(worker_task, payloads).get(
                        timeout=86400,
                    )
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
                    sys.exit(1)

                successful = [
                    result for result in results if not result.get('failed', False)
                ]
                failed_count = len(results) - len(successful)
                if failed_count:
                    print(
                        f'    [!] WARNING: {failed_count} runs failed due to '
                        'deadlock, timeout, or collision (excluded).'
                    )
                warnings = sorted({
                    f"Seed {result['seed']}: {warning}"
                    for result in results
                    for warning in result.get('warnings', [])
                })
                for warning in warnings:
                    print(f'    [!] {warning}')

                metric_values = {
                    'delivery_time': [r['delivery_time'] for r in successful],
                    'makespan': [r['makespan'] for r in successful],
                    'sim_time': [r['sim_time'] for r in successful],
                    'terraforming_cost': [
                        r['terraforming_cost'] for r in successful
                    ],
                }
                for metric, samples in metric_values.items():
                    if samples:
                        metric_results[f'avg_{metric}'].append(float(np.mean(samples)))
                        metric_results[f'std_{metric}'].append(float(np.std(samples)))
                    else:
                        metric_results[f'avg_{metric}'].append(float('nan'))
                        metric_results[f'std_{metric}'].append(float('nan'))

                metric_results['raw_data'][str(value)] = {
                    'delivery_times': metric_values['delivery_time'],
                    'makespans': metric_values['makespan'],
                    'sim_times': metric_values['sim_time'],
                    'terraforming_costs': metric_values['terraforming_cost'],
                    'failed_runs': failed_count,
                    'warnings': warnings,
                }
                print(
                    f"    -> Avg DT: {metric_results['avg_delivery_time'][-1]:.2f} | "
                    f"Avg MS: {metric_results['avg_makespan'][-1]:.2f} | "
                    f"Avg Sim Time: {metric_results['avg_sim_time'][-1]:.2f}s | "
                    f"Avg TF Cost: "
                    f"{metric_results['avg_terraforming_cost'][-1]:.2f}"
                )

            scenario_results[parameter_name] = metric_results

            fixed_text = f'Base Params:\nGamma = {GAMMA}\n' + ''.join(
                f'{name.capitalize()} = {value}\n'
                for name, value in base_params.items()
                if name.capitalize() != parameter_name
            )
            plot_specs = (
                ('delivery_time', 'Average Delivery Time (Timesteps)', 'o', True),
                ('makespan', 'Average Makespan (Timesteps)', 's', True),
                ('sim_time', 'Simulation Processing Time (Seconds)', '^', False),
                ('terraforming_cost', 'Energy/Penalty Score', 'd', False),
            )
            for metric, ylabel, marker, use_baselines in plot_specs:
                save_metric_plot(
                    os.path.join(
                        scenario_dir, f'Plot_{metric}_{parameter_name}.png',
                    ),
                    scenario_name, parameter_name, sweep_values,
                    metric_results[f'avg_{metric}'],
                    metric_results[f'std_{metric}'],
                    metric, ylabel, marker, COLORS[parameter_name][metric],
                    fixed_text, baselines if use_baselines else None,
                )

        all_results['scenarios'][scenario_name] = scenario_results
        with open(
            os.path.join(scenario_dir, 'scenario_results.json'), 'w',
        ) as result_file:
            json.dump(scenario_results, result_file, indent=4, allow_nan=True)
        print(f'[+] Scenario results saved in {scenario_dir}')

    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w') as result_file:
        json.dump(all_results, result_file, indent=4, allow_nan=True)
    print(f'All results saved in {results_path}')


if __name__ == '__main__':
    main()
