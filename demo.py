import argparse
import yaml
import json
import os
from Simulation.TP_with_recovery import TokenPassingRecovery
import RoothPath
from Simulation.tasks_and_delays_maker import *
from Simulation.simulation_new_recovery import SimulationNewRecovery
import subprocess
import sys
import random
import Simulation.cell_value_predictors as cvp

SELECTED_PREDICTOR = cvp.frequency_predictor

def get_cell_value_grid(tp, dimensions):
    return [
        [tp.cell_values[(x, y)] for x in range(dimensions[0])]
        for y in range(dimensions[1])
    ]


def snapshot_agent_states(simulation, tp):
    token = tp.get_token()
    states = {}
    for agent_name, actual_path in simulation.actual_paths.items():
        position = [actual_path[-1]['x'], actual_path[-1]['y']]
        planned_path = token['agents'][agent_name]
        assignment = token['agents_to_tasks'].get(agent_name)
        if agent_name in simulation.delayed_agents:
            status = 'Delayed'
        elif len(planned_path) > 1:
            if assignment and assignment['task_name'] == 'safe_idle':
                status = 'Moving to safe position'
            elif assignment:
                status = 'Moving'
            else:
                status = 'Recovering'
        elif assignment and assignment['task_name'] == 'safe_idle':
            status = 'At safe position'
        elif assignment and tuple(assignment['goal']) == tuple(position):
            status = 'Completed'
        elif assignment:
            status = 'Waiting'
        else:
            status = 'Idle'

        state = {
            'status': status,
            'planned_path': [
                list(position) for position in token['agents'].get(agent_name, [])
            ],
        }
        if assignment:
            state['task_name'] = assignment['task_name']
            state['start'] = list(assignment['start'])
            state['goal'] = list(assignment['goal'])
        states[agent_name] = state
    return {'time': simulation.get_time(), 'agents': states}


def advance_and_record_delivery_times(
        simulation, tp, task_locations, delivery_start_times, delivery_goal_times):
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
        if task_name not in delivery_start_times and (
                newly_assigned and previous_positions[agent_name] == task_start):
            delivery_start_times[task_name] = current_time

        for step in path[previous_path_lengths[agent_name]:]:
            position = (step['x'], step['y'])
            if task_name not in delivery_start_times and position == task_start:
                delivery_start_times[task_name] = step['t']
            if (task_name in delivery_start_times
                    and task_name not in delivery_goal_times
                    and position == task_goal):
                delivery_goal_times[task_name] = step['t']


def calculate_average_delivery_time(
        completed_task_names, delivery_start_times, delivery_goal_times):
    missing_times = [
        task_name for task_name in completed_task_names
        if task_name not in delivery_start_times
        or task_name not in delivery_goal_times
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


if __name__ == '__main__':
    random.seed(17)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', help='Robustness parameter for k-TP', default=None, type=int)
    parser.add_argument('-p', help='Robustness parameter for p-TP', default=None, type=float)
    parser.add_argument('-pd', help='Expected probability of an agent of being delayed at any time step (p-TP)',
                        default=0, type=float)
    parser.add_argument('-p_iter', help='Number of times a new path can be recalculated if the one calculated '
                                        'before exceeds the probability threshold (p-TP)',
                        default=1, type=int)
    parser.add_argument('-a_star_max_iter', help='Maximum number of states explored by the low-level algorithm',
                        default=5000, type=int)
    parser.add_argument('-slow_factor', help='Slow factor of visualization', default=1, type=int)
    parser.add_argument('-not_rand', help='Use if input has fixed tasks and delays', action='store_true')
    parser.add_argument('-radius', help="Terraforming radius (Radius k)", default=1, type=int)
    parser.add_argument('-alpha', help="Weight alpha", default=100, type=float)
    parser.add_argument('-gamma', help="Weight gamma for obstacle configuration cost", default=0, type=float)
    parser.add_argument('-laziness', help="Laziness of the radar (number of steps it can take in a single time step)", default=1, type=int)
    parser.add_argument('-no_vis', help='Disable final visualization', action='store_true')

    args = parser.parse_args()

    if args.k is None:
        args.k = 0
    if args.p is None:
        args.p = 1

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
        config = json.load(json_file)
    args.param = os.path.join(RoothPath.get_root(), os.path.join(config['input_path'], config['input_name']))
    args.output = os.path.join(RoothPath.get_root(), 'output.yaml')

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', []) 
    non_task_endpoints = param['map']['non_task_endpoints']
    agents = param['agents']
    if args.not_rand:
        # Old fixed tasks and delays
        tasks = param['tasks']
        delays = param['delays']
    else:
        # Generate random tasks and delays
        tasks, delays = gen_tasks_and_delays(agents, param['map']['start_locations'], param['map']['goal_locations'],
                                             param['n_tasks'], param['task_freq'], param['n_delays_per_agent'])
    param['tasks'] = tasks
    param['delays'] = delays
    with open(args.param + config['visual_postfix'], 'w') as param_file:
        yaml.safe_dump(param, param_file)

    # Simulate
    simulation = SimulationNewRecovery(tasks, agents, delays=delays)
    tp = TokenPassingRecovery(agents, dimensions, obstacles, non_task_endpoints, simulation,
                              a_star_max_iter=args.a_star_max_iter, k=args.k,
                              replan_every_k_delays=False, pd=args.pd, p_max=args.p, p_iter=args.p_iter, terraforming_radius=args.radius,
                              new_recovery=True, movable_obstacles=movable_obstacles, alpha=args.alpha, laziness=args.laziness,
                              cell_value_predictor=SELECTED_PREDICTOR, gamma=args.gamma)
    cell_values_history = [{
        'time': simulation.get_time(),
        'values': get_cell_value_grid(tp, dimensions),
    }]
    agent_state_history = [snapshot_agent_states(simulation, tp)]
    task_locations = {
        task['task_name']: (tuple(task['start']), tuple(task['goal']))
        for task in tasks
    }
    delivery_start_times = {}
    delivery_goal_times = {}
    failure_message = None
    try:
        while tp.get_completed_tasks() != len(tasks):
            advance_and_record_delivery_times(
                simulation, tp, task_locations,
                delivery_start_times, delivery_goal_times,
            )
            current_values = get_cell_value_grid(tp, dimensions)
            if current_values != cell_values_history[-1]['values']:
                cell_values_history.append({
                    'time': simulation.get_time(),
                    'values': current_values,
                })
            agent_state_history.append(snapshot_agent_states(simulation, tp))
    except SystemExit as exc:
        failure_message = str(exc) or 'Simulation stopped with a non-zero exit status.'
        print(f"Simulation failed at time {simulation.get_time()}: {failure_message}")
        current_values = get_cell_value_grid(tp, dimensions)
        if current_values != cell_values_history[-1]['values']:
            cell_values_history.append({
                'time': simulation.get_time(),
                'values': current_values,
            })
        agent_state_history.append(snapshot_agent_states(simulation, tp))

    agent_cost = 0
    p_total = 0

    
    for agent_name, path in simulation.actual_paths.items():
        if len(path) > 1:
            final_pos = (path[-1]['x'], path[-1]['y'])
            t_cost = 0
            for step in path:
                if step['x'] != final_pos[0] or step['y'] != final_pos[1]:
                    t_cost = step['t'] + 1
            agent_cost += t_cost

    
    for obs_id, path in tp.get_token()['obstacles_paths'].items():
        for i in range(1, len(path)):
            if path[i]['x'] != path[i-1]['x'] or path[i]['y'] != path[i-1]['y']:
                p_total += 1
                
    obstacle_config_cost = sum(
        tp.cell_values[(path[-1]['x'], path[-1]['y'])]
        for path in tp.get_token()['obstacles_paths'].values()
        if path
    )
    cost = agent_cost + (tp.alpha * p_total) + (tp.gamma * obstacle_config_cost)
    completed_times = tp.get_completed_tasks_times()
    final_makespan = max(completed_times.values()) if completed_times else 0
    average_delivery_time = calculate_average_delivery_time(
        completed_times, delivery_start_times, delivery_goal_times,
    )
    terraforming_events = [
        {
            'time': current['t'],
            'obstacle': obstacle_name,
        }
        for obstacle_name, path in tp.get_token()['obstacles_paths'].items()
        for previous, current in zip(path, path[1:])
        if (previous['x'], previous['y']) != (current['x'], current['y'])
    ]
    terraforming_events.sort(key=lambda event: event['time'])
    combined_schedule = simulation.actual_paths.copy()
    max_time = simulation.get_time()
    cleaned_obs_paths = {}
    for obs_id, path in tp.get_token()['obstacles_paths'].items():
        time_map = {}
        for step in path:
            if step['t'] not in time_map:
                time_map[step['t']] = step
        continuous_path = []
        last_pos = path[0] if path else {'x': 0, 'y': 0}
        for t in range(max_time + 1):
            if t in time_map:
                last_pos = time_map[t]
            continuous_path.append({'t': t, 'x': last_pos['x'], 'y': last_pos['y']})
        cleaned_obs_paths[obs_id] = continuous_path
    combined_schedule.update(cleaned_obs_paths)
    output = {'schedule': combined_schedule, 'cost': cost,
              'completed_tasks_times': completed_times,
              'n_replans': tp.get_n_replans(),
              'cell_values': get_cell_value_grid(tp, dimensions),
              'cell_values_history': cell_values_history,
              'agent_state_history': agent_state_history,
              'final_makespan': final_makespan,
              'average_delivery_time': average_delivery_time,
              'terraforming_events': terraforming_events,
              'terraforming_alpha': tp.alpha,
              'demo_settings': {
                  'alpha': tp.alpha,
                  'terraforming_radius': args.radius,
                  'laziness': args.laziness,
                  'map_name': os.path.basename(args.param),
                  'predictor': getattr(
                      SELECTED_PREDICTOR,
                      '__name__',
                      type(SELECTED_PREDICTOR).__name__,
                  ),
              },
              'failed': failure_message is not None,
              'failure_message': failure_message}
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

    if not args.no_vis:
        create = [sys.executable, '-m', 'Utils.Visualization.visualize_interactive',
                  '-map', args.param + config['visual_postfix'],
                  '-schedule', args.output, '-slow_factor', str(args.slow_factor)]
        subprocess.call(create)

    if failure_message is not None:
        sys.exit(1)
