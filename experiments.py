import os
import json
import yaml
import random
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import sys
import multiprocessing
import io
from collections import defaultdict

import RoothPath
from Utils.pool_with_subprocess import PoolWithSubprocess
from Simulation.simulation_new_recovery import SimulationNewRecovery
from Simulation.TP_with_recovery import TokenPassingRecovery
from Simulation.tasks_and_delays_maker import gen_tasks_and_delays

NUM_RUNS = 25
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

SCENARIOS = {
    "Scenario_1_Heavy_World": {"alpha": 20.0, "radius": 2, "laziness": 3},
    "Scenario_2_Highly_Malleable": {"alpha": 0.5, "radius": 5, "laziness": 3},
    "Scenario_3_Strictly_Reactive": {"alpha": 1.0, "radius": 3, "laziness": 1},
    "Scenario_4_Topologically_Constrained": {"alpha": 1.0, "radius": 1, "laziness": 5}
}

SWEEP_VALUES = {
    "Alpha": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10, 20, 100],
    "Radius": [1, 2, 3, 4, 5, 6],
    "Laziness": [1, 2, 3, 4, 5]
}

class LogCatcher:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        
    def get_critical_warnings(self):
        output = self.captured_output.getvalue()
        warnings = set()
        keywords = ['fatal exception']
        
        for line in output.split('\n'):
            line = line.strip()
            if any(kw in line.lower() for kw in keywords):
                warnings.add(line)
        return list(warnings)

def verify_no_collisions(simulation, tp):
    states_dict = defaultdict(int)
    full_paths = {}
    
    for agent_name, path in simulation.actual_paths.items():
        full_paths[agent_name] = path
        
    obstacles_paths = tp.get_token()['obstacles_paths']
    max_time = simulation.get_time()
    
    for obs_id, path in obstacles_paths.items():
        continuous_path = []
        path_dict = {step['t']: step for step in path}
        if path:
            last_pos = path[0]
        else:
            last_pos = {'x': 0, 'y': 0}
            
        for t in range(max_time + 1):
            if t in path_dict:
                last_pos = path_dict[t]
            continuous_path.append({'t': t, 'x': last_pos['x'], 'y': last_pos['y']})
        full_paths[obs_id] = continuous_path

    for entity, path in full_paths.items():
        for step in path:
            state_key = f"{step['t']}_{step['x']}_{step['y']}"
            states_dict[state_key] += 1
            if states_dict[state_key] > 1:
                return True
    return False

def generate_map_preview(yaml_path, output_filename):
    with open(yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    dims = map_data['map']['dimensions']
    static_obs = map_data['map'].get('obstacles', [])
    movable_obs = map_data['map'].get('movable_obstacles', [])
    endpoints = map_data['map'].get('non_task_endpoints', [])
    agents = map_data.get('agents', [])

    fig, ax = plt.subplots(figsize=(dims[0], dims[1]))
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_xticks(range(dims[0] + 1))
    ax.set_yticks(range(dims[1] + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_aspect('equal')

    def draw_square(coords, color, edgecolor='black', hatch=None, zorder=1):
        x, y = coords[0], coords[1]
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=edgecolor, hatch=hatch, zorder=zorder))

    for ep in endpoints:
        draw_square(ep, '#cce5ff', edgecolor='#0066cc', zorder=1)
        ax.text(ep[0] + 0.5, ep[1] + 0.5, 'EP', color='#0066cc', ha='center', va='center', fontsize=10, fontweight='bold')

    for obs in static_obs:
        draw_square(obs, '#333333', zorder=2)

    for mobs in movable_obs:
        draw_square(mobs, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    for agent in agents:
        start = agent['start']
        cx, cy = start[0] + 0.5, start[1] + 0.5
        ax.add_patch(patches.Circle((cx, cy), 0.35, facecolor='#0033cc', edgecolor='black', zorder=4))
        label = agent['name'].replace('agent', 'A')
        ax.text(cx, cy, label, color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)

    legend_elements = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(facecolor='#ff9900', edgecolor='#cc0000', hatch='///', label='Movable Obstacle'),
        patches.Patch(facecolor='#cce5ff', edgecolor='#0066cc', label='Endpoint (Task Area)'),
        Line2D([0], [0], marker='o', color='w', label='Agent', markerfacecolor='#0033cc', markersize=12, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    plt.title(f"Map Preview", fontsize=16, pad=15)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')

def worker_task(payload):
    yaml_path = payload['yaml_path']
    laziness = payload['laziness']
    radius = payload['radius']
    alpha = payload['alpha']
    seed = payload['seed']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    agents = param['agents']
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']

    n_tasks = param.get('n_tasks', 20)
    task_freq = param.get('task_freq', 1.0)
    n_delays_per_agent = param.get('n_delays_per_agent', 0)

    random.seed(seed)
    
    tasks, delays = gen_tasks_and_delays(
        agents,
        param['map']['start_locations'],
        param['map']['goal_locations'],
        n_tasks,
        task_freq,
        n_delays_per_agent
    )

    with LogCatcher() as catcher:
        simulation = SimulationNewRecovery(tasks, agents, delays=delays)
        tp = TokenPassingRecovery(
            agents, dimensions, obstacles, non_task_endpoints, simulation,
            a_star_max_iter=ASTAR_MAX_ITER,
            k=0, pd=0.0, p_max=1, p_iter=1,
            terraforming_radius=radius,
            new_recovery=True,
            movable_obstacles=movable_obstacles,
            alpha=alpha,
            laziness=laziness
        )

        while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < MAX_SIM_TIME:
            simulation.time_forward(tp)

        warnings = catcher.get_critical_warnings()

        if simulation.get_time() >= MAX_SIM_TIME:
            warnings.append(f"TIMEOUT: MAX_SIM_TIME reached.")
            return {
                'makespan': None,
                'service_time': None,
                'seed': seed,
                'failed': True,
                'warnings': warnings
            }
            
        has_collision = verify_no_collisions(simulation, tp)
        if has_collision:
            warnings.append("PHYSICAL COLLISION DETECTED DURING VALIDATION!")
            return {
                'makespan': None,
                'service_time': None,
                'seed': seed,
                'failed': True,
                'warnings': warnings
            }

        completed_times = tp.get_completed_tasks_times()
        task_starts = {t['task_name']: t['start_time'] for t in tasks}

        makespan = max(completed_times.values()) if completed_times else 0

        serv_times = []
        for t_name, c_time in completed_times.items():
            serv_times.append(c_time - task_starts[t_name])
            
        avg_service_time = sum(serv_times) / len(serv_times) if serv_times else 0

    return {
        'makespan': makespan,
        'service_time': avg_service_time,
        'seed': seed,
        'failed': False,
        'warnings': warnings
    }

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" STARTING MULTI-SCENARIO SENSITIVITY ANALYSIS")
    print("="*50)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)
        
    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)

    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots1', map_name_clean)
    os.makedirs(base_plots_dir, exist_ok=True)
    
    map_img_path = os.path.join(base_plots_dir, 'map_preview.png')
    generate_map_preview(map_path, map_img_path)
    
    print(f"Test Map: {map_yaml_name}")
    print(f"Base Directory: {base_plots_dir}")
    print(f"Map Preview Generated: {map_img_path}\n")

    cores = max(1, multiprocessing.cpu_count() - 2)
    colors_st = {'Alpha': '#d62728', 'Radius': '#1f77b4', 'Laziness': '#2ca02c'}
    colors_ms = {'Alpha': '#ff7f0e', 'Radius': '#9467bd', 'Laziness': '#8c564b'}

    for scen_name, base_params in SCENARIOS.items():
        print(f">>> Executing Scenario: {scen_name} <<<")
        
        scen_dir = os.path.join(base_plots_dir, scen_name)
        os.makedirs(scen_dir, exist_ok=True)
        
        master_results = {}

        for param_name, sweep_vals in SWEEP_VALUES.items():
            print(f"--- Sweeping Parameter: {param_name} ---")
            
            master_results[param_name] = {
                'values': sweep_vals,
                'base_settings': base_params.copy(),
                'avg_service_time': [],
                'std_service_time': [],
                'avg_makespan': [],
                'std_makespan': [],
                'raw_data': {}
            }

            for val in sweep_vals:
                print(f"  Testing {param_name} = {val} | Parallel Runs: {NUM_RUNS}...")
                
                payloads = []
                for seed in range(1, NUM_RUNS + 1):
                    payload = {
                        'yaml_path': map_path,
                        'seed': seed,
                        'alpha': val if param_name == 'Alpha' else base_params['alpha'],
                        'radius': val if param_name == 'Radius' else base_params['radius'],
                        'laziness': val if param_name == 'Laziness' else base_params['laziness']
                    }
                    payloads.append(payload)

                pool = PoolWithSubprocess(processes=cores)
                try:
                    
                    results = pool.map_async(worker_task, payloads).get(timeout=86400)
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C premuto! Terminazione forzata dei processi in corso...")
                    pool.terminate() 
                    pool.join()
                    sys.exit(1)   

                successful_results = [r for r in results if not r.get('failed', False)]
                failed_count = len(results) - len(successful_results)
                
                all_warnings = []
                for r in results:
                    if r.get('warnings'):
                        for w in r['warnings']:
                            all_warnings.append(f"Seed {r['seed']}: {w}")
                
                if failed_count > 0:
                    print(f"      [!] WARNING: {failed_count} runs ended in deadlock or collision (excluded).")
                    
                if all_warnings:
                    print(f"      [!] CRITICAL LOGS DETECTED:")
                    for w in sorted(list(set(all_warnings))): 
                        print(f"          - {w}")

                if successful_results:
                    service_times = [r['service_time'] for r in successful_results]
                    makespans = [r['makespan'] for r in successful_results]

                    mean_st = np.mean(service_times)
                    std_st = np.std(service_times)
                    mean_ms = np.mean(makespans)
                    std_ms = np.std(makespans)
                else:
                    mean_st = std_st = mean_ms = std_ms = 0
                    service_times = []
                    makespans = []

                master_results[param_name]['avg_service_time'].append(mean_st)
                master_results[param_name]['std_service_time'].append(std_st)
                master_results[param_name]['avg_makespan'].append(mean_ms)
                master_results[param_name]['std_makespan'].append(std_ms)
                master_results[param_name]['raw_data'][str(val)] = {
                    'service_times': service_times,
                    'makespans': makespans
                }

                print(f"    -> Avg ST: {mean_st:.2f} (+/-{std_st:.2f}) | Avg MS: {mean_ms:.2f} (+/-{std_ms:.2f})")

        json_path = os.path.join(scen_dir, 'scenario_results.json')
        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        for param_name, data in master_results.items():
            x = data['values']
            y_st = data['avg_service_time']
            yerr_st = data['std_service_time']
            y_ms = data['avg_makespan']
            yerr_ms = data['std_makespan']
            
            fixed_text = "Base Params:\n"
            for k, v in data['base_settings'].items():
                if k.capitalize() != param_name:
                    fixed_text += f"{k.capitalize()} = {v}\n"

            plt.figure(figsize=(8, 6))
            plt.errorbar(x, y_st, yerr=yerr_st, fmt='-o', color=colors_st[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, 
                         markersize=8, label='Avg Service Time')
            plt.title(f'{scen_name}: Service Time vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Average Service Time (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            plt.figure(figsize=(8, 6))
            plt.errorbar(x, y_ms, yerr=yerr_ms, fmt='-s', color=colors_ms[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, 
                         markersize=8, label='Avg Makespan')
            plt.title(f'{scen_name}: Makespan vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Average Makespan (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

        print(f"[+] Scenario {scen_name} completed. Data saved in {scen_dir}\n")

    print("==================================================")
    print(" ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print("==================================================\n")