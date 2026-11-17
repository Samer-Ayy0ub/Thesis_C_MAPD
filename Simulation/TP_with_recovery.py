"""
Python implementation of Token Passing algorithms to solve MAPD problems with delays
author: Giacomo Lodigiani (@Lodz97)
"""
from math import fabs
import random
from Simulation.CBS.cbs import CBS, Environment
from Simulation.cell_value_predictors import frequency_predictor, manhattan_task_predictor, null_predictor
from Simulation.markov_chains import MarkovChainsMaker
from collections import defaultdict

MAX_FAILED_DEADLOCK_RECOVERY_ATTEMPTS = 1000
RECOVERY_RADIUS = 4

class TokenPassingRecovery(object):
    def __init__(self, agents, dimesions, obstacles, non_task_endpoints, simulation, a_star_max_iter=4000, k=0,
                 replan_every_k_delays=False, pd=None, p_max=1, p_iter=1, new_recovery=False, movable_obstacles=None, alpha=0.5, terraforming_radius=3, laziness=1, cell_value_predictor=None, gamma=0.0):
        self.agents = agents
        self.laziness = laziness
        self.dimensions = dimesions
        self.cell_visit_counts = {
            (x, y): 0
            for x in range(dimesions[0])
            for y in range(dimesions[1])
        }
        self.obstacles = set(tuple(o) for o in obstacles)
        self.movable_obstacles = set(tuple(o) for o in (movable_obstacles or []))
        self.non_task_endpoints = non_task_endpoints
        if len(agents) > len(non_task_endpoints):
            print('There are more agents than non task endpoints, instance is not well-formed.')
            exit(1)
        self.v_ep = set(tuple(ep) for ep in self.non_task_endpoints)
        for t in simulation.tasks:
            self.v_ep.add(tuple(t['start']))
            self.v_ep.add(tuple(t['goal']))
        # TODO: Check all properties for well-formedness
        self.token = {}
        self.simulation = simulation
        self.a_star_max_iter = a_star_max_iter
        self.k = k
        self.p_max = p_max
        self.alpha = alpha
        self.gamma = gamma
        self.movable_obstacles_list = movable_obstacles or []
        self.terraforming_radius = terraforming_radius
        self.cell_value_predictor = cell_value_predictor or null_predictor
        if self.k != 0 and self.p_max != 1:
            print('Use of k and p robustness at same time not allowed.')
            exit(-1)
        if self.k < 0:
            print('k should be >= 0.')
            exit(1)
        if self.k == 0 and not new_recovery:
            print('k = 0 not supported for this recovery type.')
            exit(1)
        self.replan_every_k_delays = replan_every_k_delays
        if k == 0 and replan_every_k_delays:
            print('If k = 0 replan_every_k_delays must be False. Setting to False...')
            self.replan_every_k_delays = False
        if self.k == 0:
            if (self.p_max < 0 or self.p_max > 1):
                print('Max conflict probability must be between 0 and 1.')
                exit(1)
            self.pd = pd
            if self.p_max != 1 and not self.pd:
                print('To use p robustness need to set pd.')
                exit(1)
            if self.pd and (pd < 0 or pd > 1):
                print('Probability of delay must be between 0 and 1.')
                exit(1)
            self.p_iter = p_iter
            if self.p_iter <= 0:
                print('p_iter should be > 0.')
                exit(1)
        else:
            self.p_max = 1
            self.pd = None
            self.p_iter = 1
        self.new_recovery = new_recovery
        self.init_token()
        self.update_cell_values()
        self.check_initial_well_formedness()

    def init_token(self):
        self.token['agents'] = {}
        self.token['tasks'] = {}
        self.token['obstacles_paths'] = {}
        self.token['original_obs_pos'] = {}
        for i, pos in enumerate(self.movable_obstacles_list):
            obs_name = f"obs_{i}"
            self.token['obstacles_paths'][obs_name] = [{'t': 0, 'x': pos[0], 'y': pos[1], 'agent': None}]
            self.token['original_obs_pos'][obs_name] = tuple(pos)
        self.token['start_tasks_times'] = {}
        self.token['completed_tasks_times'] = {}
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        self.token['agents_to_tasks'] = {}
        self.token['completed_tasks'] = 0
        self.token['n_replans'] = 0
        self.token['path_ends'] = set()
        self.token['occupied_non_task_endpoints'] = set()
        self.token['delayed_agents'] = []
        self.token['delayed_agents_to_reach_task_start'] = []
        self.token['agent_at_end_path'] = []
        self.token['agent_at_end_path_pos'] = []
        self.token['agents_in_recovery_trial'] = []
        for a in self.agents:
            self.token['agents'][a['name']] = [a['start']]
            self.token['path_ends'].add(tuple(a['start']))
        self.token['prob_exceeded'] = False
        self.token['deadlock_count_per_agent'] = defaultdict(lambda: 0)
        self.token['deadlock_recovery_failures_per_agent'] = defaultdict(lambda: 0)
        self.token['deadlock_recovery_last_pos_per_agent'] = {}

    def check_initial_well_formedness(self):
        if len(self.v_ep) <= 1:
            return
            
        import sys
        import collections

        ep_tuples = set(tuple(ep) for ep in self.v_ep)
        movable_obs_set = set(tuple(o) for o in self.movable_obstacles_list)
        
        for obs in movable_obs_set:
            if obs in ep_tuples:
                print(f"\nFATAL ERROR: Movable obstacle spawned directly on endpoint {obs}!")
                sys.exit(1)
                
        def check_connectivity(blocked_cells):
            start_ep = next(iter(ep_tuples))
            if start_ep in blocked_cells: return set()
            visited = {start_ep}
            reached = {start_ep}
            queue = collections.deque([start_ep])
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if len(reached) == len(ep_tuples): return reached
            while queue:
                curr = queue.popleft()
                for dx, dy in moves:
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if 0 <= nxt[0] < self.dimensions[0] and 0 <= nxt[1] < self.dimensions[1]:
                        if nxt not in blocked_cells and nxt not in visited:
                            visited.add(nxt)
                            if nxt in ep_tuples:
                                reached.add(nxt)
                                if len(reached) == len(ep_tuples): return reached
                            else:
                                queue.append(nxt)
            return reached

        # 1. Controllo Rigoroso Iniziale
        initial_reached = check_connectivity(self.obstacles | movable_obs_set)
        if len(initial_reached) == len(ep_tuples):
            return

        # 2. Controllo Multi-Ostacolo: La mappa base (senza ostacoli mobili) è connessa?
        base_reached = check_connectivity(self.obstacles)
        if len(base_reached) != len(ep_tuples):
            print("\n" + "="*70)
            print("FATAL ERROR: The STATIC environment is structurally broken.")
            print(f"Missing endpoints even if all movable obstacles are removed: {ep_tuples - base_reached}")
            print("="*70 + "\n")
            sys.exit(1)
            
        # 3. Controllo Spazio di Parcheggio: Verifichiamo che ogni carrello possa uscire dalla corsia
        radius = self.terraforming_radius
        for curr_pos in movable_obs_set:
            has_spot = False
            for dx in range(-radius, radius + 1):
                for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
                    v_park = (curr_pos[0] + dx, curr_pos[1] + dy)
                    if v_park == curr_pos: continue
                    if not (0 <= v_park[0] < self.dimensions[0] and 0 <= v_park[1] < self.dimensions[1]): continue
                    if v_park in self.obstacles: continue
                    if v_park in ep_tuples: continue
                    has_spot = True
                    break
                if has_spot: break
                
            if not has_spot:
                print("\n" + "="*70)
                print(f"FATAL ERROR: Obstacle at {curr_pos} is permanently trapped!")
                print(f"There is no valid parking spot within terraforming radius {radius}.")
                print("="*70 + "\n")
                sys.exit(1)
                
        # Se passiamo questi test, il Supermercato è pronto ad aprire!
        return

    def get_idle_agents(self):
        agents = {}
        for name, path in self.token['agents'].items():
            if len(path) == 1:
                agents[name] = path
        return agents
    
    # def get_current_movable_obstacles(self):
    #     current_obs = []
    #     base_time = self.simulation.get_time() - 1
    #     for i in range(len(self.movable_obstacles_list)):
    #         obs_id = f"obs_{i}"
    #         path = self.token['obstacles_paths'][obs_id]
    #         current_pos = path[0]
    #         for step in path:
    #             if step['t'] <= base_time:
    #                 current_pos = step
    #             else:
    #                 break
    #         current_obs.append((current_pos['x'], current_pos['y']))
    #     return current_obs
    def get_current_movable_obstacles(self):
        timelines = {}
        abs_time_start = self.simulation.get_time() - 1
        for i in range(len(self.movable_obstacles_list)):
            obs_id = f"obs_{i}"
            path = self.token['obstacles_paths'][obs_id]
            
            valid_steps = [step for step in path if step['t'] <= abs_time_start]
            base_pos = max(valid_steps, key=lambda s: s['t']) if valid_steps else ({'x': 0, 'y': 0} if not path else path[0])
                    
            rel_path = [{'t': 0, 'x': base_pos['x'], 'y': base_pos['y']}]
            
            for step in path:
                if step['t'] > abs_time_start:
                    rel_path.append({'t': step['t'] - abs_time_start, 'x': step['x'], 'y': step['y']})
                    
            timelines[obs_id] = rel_path
        return timelines

    def admissible_heuristic(self, task_pos, agent_pos):
        return fabs(task_pos[0] - agent_pos[0]) + fabs(task_pos[1] - agent_pos[1])

    def get_closest_task_name(self, available_tasks, agent_pos):
        closest = random.choice(list(available_tasks.keys()))
        dist = self.admissible_heuristic(available_tasks[closest][0], agent_pos)
        for task_name, task in available_tasks.items():
            if self.admissible_heuristic(task[0], agent_pos) < dist:
                closest = task_name
        return closest

    def get_moving_obstacles_agents(self, agents, time_start):
        obstacles = {}
        current_sim_time = self.simulation.get_time()
        abs_time_start = (current_sim_time - 1) + time_start
        base_time = current_sim_time - 1
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    obstacles[(path[i][0], path[i][1], k)] = name
                    for j in range(1, self.k + 1):
                        if i - j >= time_start:
                            obstacles[(path[i][0], path[i][1], k - j)] = name
                        obstacles[(path[i][0], path[i][1], k + j)] = name
                    # Mark last element with negative time to later turn it into idle obstacle
                    if i == len(path) - 1:
                        obstacles[(path[i][0], path[i][1], -k)] = name
        
        def get_obs_pos_at(path, target_t):
            valid_steps = [s for s in path if s['t'] <= target_t]
            if not valid_steps:
                return (path[0]['x'], path[0]['y']) if path else (0,0)
            best = max(valid_steps, key=lambda s: s['t'])
            return (best['x'], best['y'])

        for obs_id, path in self.token['obstacles_paths'].items():
            if not path:
                continue
            max_t = max(step['t'] for step in path)
            for t in range(abs_time_start, max_t + 2):
                curr_pos = get_obs_pos_at(path, t)
                prev_pos = get_obs_pos_at(path, t - 1)
                next_pos = get_obs_pos_at(path, t + 1)
                
                if curr_pos != prev_pos or curr_pos != next_pos:
                    relative_t = t - abs_time_start
                    if relative_t >= 0:
                        obstacles[(curr_pos[0], curr_pos[1], relative_t)] = obs_id
                        for j in range(1, self.k + 1):
                            if relative_t - j >= 0:
                                obstacles[(curr_pos[0], curr_pos[1], relative_t - j)] = obs_id
                            obstacles[(curr_pos[0], curr_pos[1], relative_t + j)] = obs_id
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, delayed_agents, time_start):
        obstacles = set()
        for path in agents_paths:
            if len(path) == 1:
                obstacles.add((path[0][0], path[0][1]))
            if 1 < len(path) <= time_start:
                obstacles.add((path[-1][0], path[-1][1]))
        for agent_name in delayed_agents:
            obstacles.add(tuple(self.token['agents'][agent_name][0]))
        return obstacles

    def check_safe_idle(self, agent_pos):
        if tuple(agent_pos) not in self.v_ep:
            return False
        for task_name, task in self.token['tasks'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    # def get_closest_non_task_endpoint(self, agent_pos):
    #     dist = -1
    #     res = -1
    #     for endpoint in self.non_task_endpoints:
    #         if tuple(endpoint) not in self.token['occupied_non_task_endpoints']:
    #             if dist == -1:
    #                 dist = self.admissible_heuristic(endpoint, agent_pos)
    #                 res = endpoint
    #             else:
    #                 tmp = self.admissible_heuristic(endpoint, agent_pos)
    #                 if tmp < dist:
    #                     dist = tmp
    #                     res = endpoint
    #     if res == -1:
    #         print('Error in finding non-task endpoint, is instance well-formed?')
    #         exit(1)
    #     return res

    def get_closest_non_task_endpoint(self, agent_pos):
        dist = -1
        res = -1
        
        active_task_nodes = set()
        for task in self.token['tasks'].values():
            active_task_nodes.add(tuple(task[0]))
            active_task_nodes.add(tuple(task[1]))
        for task in self.token['agents_to_tasks'].values():
            active_task_nodes.add(tuple(task['start']))
            active_task_nodes.add(tuple(task['goal']))

        
        for endpoint in self.non_task_endpoints:
            ep_tuple = tuple(endpoint)
            if ep_tuple not in self.token['occupied_non_task_endpoints'] and ep_tuple not in active_task_nodes:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
                        
        
        if res == -1:
            for endpoint in self.non_task_endpoints:
                if tuple(endpoint) not in self.token['occupied_non_task_endpoints']:
                    return endpoint
            return agent_pos # Estremo fallback
            
        return res

    def update_ends(self, agent_pos):
        if tuple(agent_pos) in self.token['path_ends']:
            self.token['path_ends'].remove(tuple(agent_pos))
        elif tuple(agent_pos) in self.token['occupied_non_task_endpoints']:
            self.token['occupied_non_task_endpoints'].remove(tuple(agent_pos))

    def record_path_visits(self, path):
        visits_recorded = False
        for previous, current in zip(path, path[1:]):
            current_cell = tuple(current)
            if current_cell != tuple(previous):
                self.cell_visit_counts[current_cell] += 1
                visits_recorded = True

        self.update_cell_values()

    def update_cell_values(self):
        context = {
            'dimensions': tuple(self.dimensions),
            'unfinished_tasks': self.get_unassigned_tasks(),
            'agents': self.agents,
            'obstacles': frozenset(self.obstacles),
            'movable_obstacles': tuple(self.movable_obstacles_list),
            'simulation_time': self.simulation.get_time(),
        }
        predicted_values = self.cell_value_predictor(
            dict(self.cell_visit_counts), context
        )
        missing_cells = self.cell_visit_counts.keys() - predicted_values.keys()
        if missing_cells:
            raise ValueError(
                f'Cell-value predictor omitted grid cells: {sorted(missing_cells)}'
            )
        self.cell_values = {
            cell: predicted_values[cell]
            for cell in self.cell_visit_counts
        }

    def get_unassigned_tasks(self):
        """Return revealed tasks that have not yet been assigned to an agent."""
        return [
            {
                'task_name': task_name,
                'start': start,
                'goal': goal,
            }
            for task_name, (start, goal) in self.token.get('tasks', {}).items()
        ]

    def get_agents_to_tasks_goals(self):
        goals = set()
        for el in self.token['agents_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_agents_to_tasks_starts_goals(self):
        starts_goals = set()
        for el in self.token['agents_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_completed_tasks(self):
        return self.token['completed_tasks']

    def get_completed_tasks_times(self):
        return self.token['completed_tasks_times']

    def get_n_replans(self):
        return self.token['n_replans']

    def get_token(self):
        return self.token

    def get_k(self):
        return self.k

    def get_replan_every_k_delays(self):
        return self.replan_every_k_delays

    def search(self, cbs, agent_name, moving_obstacles_agents):
        path = None
        if self.p_max == 1:
            path = cbs.search()
        else:
            self.token['prob_exceeded'] = False
            mk = MarkovChainsMaker(self.token['agents'], self.pd)
            for iter in range(self.p_iter):
                path = cbs.search()
                tmp = []
                if path and len(path[agent_name]) > 1:
                    for el in path[agent_name]:
                        tmp.append([el['x'], el['y']])
                    dic = mk.get_conflict_prob_given_path(tmp)
                    if dic['prob'] > self.p_max:
                        self.token['prob_exceeded'] = True
                        print('Conflict probablility to high (', dic['prob'], ') replanning...')
                        path = None
                        moving_obstacles_agents[(dic['pos_max_conf'][0], dic['pos_max_conf'][1], -1)] = agent_name
                    else:
                        break
        if path:
            print(f"CBS intended plan for agent '{agent_name}': {path}")
        return path

    def go_to_closest_non_task_endpoint(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint(agent_pos)
        moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
        idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
        agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_non_task_endpoint}
        env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep,
                  a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, 
                  terraforming_radius=0, 
            static_obstacles=self.obstacles, laziness=self.laziness, cell_values=self.cell_values,
            goal_must_remain_safe=True, gamma=self.gamma)
        cbs = CBS(env)
        result = self.search(cbs, agent_name, moving_obstacles_agents)
        if not result:
            print("Solution to non-task endpoint not found for agent", agent_name, " instance is not well-formed.")
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, RECOVERY_RADIUS)
            # exit(1)
        else:
            path_to_non_task_endpoint = result['agents']
            obs_trajectories = result['obstacles']
            base_time = self.simulation.get_time() - 1
            for obs_id, traj in obs_trajectories.items():
                if len(traj) > 0:
                    first_abs_t = base_time + traj[0]['t']
                    self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                for step in traj:
                    abs_t = base_time + step['t']
                    self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
            print('No available task for agent', agent_name, ' moving to safe idling position...')
            self.update_ends(agent_pos)
            self.token['occupied_non_task_endpoints'].add(tuple(closest_non_task_endpoint))
            self.token['agents_to_tasks'][agent_name] = {'task_name': 'safe_idle', 'start': agent_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['agents'][agent_name] = []
            for el in path_to_non_task_endpoint[agent_name]:
                self.token['agents'][agent_name].append([el['x'], el['y']])
            self.record_path_visits(self.token['agents'][agent_name])

    def get_random_close_cell(self, agent_pos, r):
        while True:
            cell = (agent_pos[0] + random.choice(range(-r - 1, r + 1)), agent_pos[1] + random.choice(range(-r - 1, r + 1)))
            if cell not in self.obstacles and cell not in self.token['path_ends'] and \
                    cell not in self.token['occupied_non_task_endpoints'] \
                    and cell not in self.get_agents_to_tasks_goals() \
                    and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
                return cell

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, r):
        current_pos = tuple(agent_pos)
        last_pos = self.token['deadlock_recovery_last_pos_per_agent'].get(agent_name)
        if last_pos is not None and current_pos != last_pos:
            self.token['deadlock_recovery_failures_per_agent'][agent_name] = 0
        self.token['deadlock_recovery_last_pos_per_agent'][agent_name] = current_pos

        self.token['deadlock_count_per_agent'][agent_name] += 1
        if self.token['deadlock_count_per_agent'][agent_name] >= 5:
            self.token['deadlock_count_per_agent'][agent_name] = 0
            random_close_cell = self.get_random_close_cell(agent_pos, r)
            moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
            idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
            agent = {'name': agent_name, 'start': agent_pos, 'goal': random_close_cell}
            env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep,
                  a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, 
                  terraforming_radius=0, 
                  well_formed_radius=self.terraforming_radius,
                                static_obstacles=self.obstacles, laziness=self.laziness, cell_values=self.cell_values,
                                goal_must_remain_safe=True, gamma=self.gamma)
            cbs = CBS(env)
            result = self.search(cbs, agent_name, moving_obstacles_agents)
            if not result:
                self.token['deadlock_recovery_failures_per_agent'][agent_name] += 1
                failures = self.token['deadlock_recovery_failures_per_agent'][agent_name]
                if failures >= MAX_FAILED_DEADLOCK_RECOVERY_ATTEMPTS:
                    task = self.token['agents_to_tasks'].get(agent_name)
                    goal = task.get('goal') if task else None
                    message = (
                        f"FATAL ERROR: Deadlock recovery failed {failures} consecutive times for "
                        f"agent '{agent_name}' at simulation time {self.simulation.get_time()}, "
                        f"position {current_pos}, task goal {goal}. Stopping simulation."
                    )
                    print(message)
                    raise SystemExit(message)
                print("No solution to deadlock recovery for agent", agent_name,
                      f"({failures}/{MAX_FAILED_DEADLOCK_RECOVERY_ATTEMPTS} failed attempts); retrying later.")
            else:
                self.token['deadlock_recovery_failures_per_agent'][agent_name] = 0
                path_to_non_task_endpoint = result['agents']
                obs_trajectories = result['obstacles']
                base_time = self.simulation.get_time() - 1
                for obs_id, traj in obs_trajectories.items():
                    if len(traj) > 0:
                        first_abs_t = base_time + traj[0]['t']
                        self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                    for step in traj:
                        abs_t = base_time + step['t']
                        self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                # Don't consider this a task, so don't add to agents_to_tasks
                print('Agent', agent_name, 'causing deadlock, moving to safer position...')
                self.update_ends(agent_pos)
                self.token['agents'][agent_name] = []
                for el in path_to_non_task_endpoint[agent_name]:
                    self.token['agents'][agent_name].append([el['x'], el['y']])
                self.record_path_visits(self.token['agents'][agent_name])

    def time_forward(self):
        # Update completed tasks
        for agent_name in self.token['agents']:
            pos = self.simulation.actual_paths[agent_name][-1]
            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] != 'safe_idle':
                self.token['completed_tasks'] = self.token['completed_tasks'] + 1
                self.token['completed_tasks_times'][
                    self.token['agents_to_tasks'][agent_name]['task_name']] = self.simulation.get_time()
                self.token['agents_to_tasks'].pop(agent_name)
                self.token['deadlock_recovery_failures_per_agent'][agent_name] = 0

            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] == 'safe_idle':
                self.token['agents_to_tasks'].pop(agent_name)

        self.update_cell_values()

        # Check delayed agents and agents affected by delays
        if self.new_recovery:
            self.token['delayed_agents'] = self.simulation.get_delayed_agents()
            for name in self.token['delayed_agents']:
                print('Agent', name, 'delayed or affected by delay!')
                path = self.token['agents'][name]
                self.token['n_replans'] = self.token['n_replans'] + 1
                self.update_ends(path[-1])
                if path[0] in self.non_task_endpoints:
                    self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                else:
                    self.token['path_ends'].add(tuple(path[0]))
                if name in self.token['agents_to_tasks']:
                    if self.token['agents_to_tasks'][name]['start'] not in path:
                        self.token['delayed_agents_to_reach_task_start'].append(name)
                self.token['agents'][name] = [path[0]]
                self.scrub_future_obstacle_trajectories(name, self.simulation.get_time() - 1)
        else:
            self.token['delayed_agents'] = []
            delayed_agents_pos = []
            for name, path in self.token['agents'].items():
                actual_state = self.simulation.get_actual_paths()[name][-1]
                if path[0] != [actual_state['x'], actual_state['y']]:
                    print('Agent', name, 'delayed!')
                    # self.token['n_replans'] = self.token['n_replans'] + 1
                    self.token['delayed_agents'].append(name)
                    delayed_agents_pos.append([actual_state['x'], actual_state['y']])
                    self.update_ends(path[-1])
                    pos = tuple([actual_state['x'], actual_state['y']])
                    if pos in self.non_task_endpoints:
                        self.token['occupied_non_task_endpoints'].add(pos)
                    else:
                        self.token['path_ends'].add(pos)
                    if self.token['agents_to_tasks'][name]['start'] not in path:
                        self.token['delayed_agents_to_reach_task_start'].append(name)
                    self.token['agents'][name] = [[actual_state['x'], actual_state['y']]]
            for name, path in self.token['agents'].items():
                if name not in self.token['delayed_agents']:
                    for i in range(len(path)):
                        if path[i] in delayed_agents_pos:
                            print('Agent', name, 'affected by delay!')
                            self.token['n_replans'] = self.token['n_replans'] + 1
                            self.update_ends(path[-1])
                            if path[0] in self.non_task_endpoints:
                                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                            else:
                                self.token['path_ends'].add(tuple(path[0]))
                            if self.token['agents_to_tasks'][name]['start'] not in path:
                                self.token['delayed_agents_to_reach_task_start'].append(name)
                            self.token['agents'][name] = [path[0]]
                            break

        # TODO do this maybe only for old recovery
        # Check if somehow an agent path collides with an idle agent (may happen because of delays)
        if not self.new_recovery:
            for name, path in self.get_idle_agents().items():
                self.token['agent_at_end_path'].append(name)
                self.token['agent_at_end_path_pos'].append(path[0])
            for name, path in self.token['agents'].items():
                if name not in self.token['agent_at_end_path']:
                    for i in range(len(path)):
                        if path[i] in self.token['agent_at_end_path_pos']:
                            print('Agent', name, 'will impact end task agent, replanning...')
                            # self.update_ends(path[-1])
                            if path[0] in self.non_task_endpoints:
                                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                            else:
                                self.token['path_ends'].add(tuple(path[0]))
                            # TODO check this rare keyerror
                            if self.token['agents_to_tasks'][name]['start'] not in path:
                                self.token['delayed_agents_to_reach_task_start'].append(name)
                            self.token['agents'][name] = [path[0]]
                            break
            self.token['agent_at_end_path'] = []
            self.token['agent_at_end_path_pos'] = []

        # Collect new tasks and assign them, if possible
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        self.update_cell_values()
        idle_agents = self.get_idle_agents()
        while len(idle_agents) > 0:
            agent_name = random.choice(list(idle_agents.keys()))
            # agent_name = list(idle_agents.keys())[0]
            all_idle_agents = self.token['agents'].copy()
            all_idle_agents.pop(agent_name)
            all_delayed_agents = self.token['delayed_agents'].copy()
            if agent_name in all_delayed_agents:
                all_delayed_agents.remove(agent_name)
            agent_pos = idle_agents.pop(agent_name)[0]
            available_tasks = {}
            for task_name, task in self.token['tasks'].items():
                if tuple(task[0]) not in self.token['path_ends'].difference({tuple(agent_pos)}) and tuple(
                        task[1]) not in self.token['path_ends'].difference({tuple(agent_pos)}) \
                        and tuple(task[0]) not in self.get_agents_to_tasks_goals() and tuple(
                    task[1]) not in self.get_agents_to_tasks_goals():
                    available_tasks[task_name] = task
            if len(available_tasks) > 0 or agent_name in self.token['agents_to_tasks']:
                if agent_name in self.token['agents_to_tasks']:
                    closest_task_name = self.token['agents_to_tasks'][agent_name]['task_name']
                    if agent_name in self.token['delayed_agents_to_reach_task_start']:
                        closest_task = [self.token['agents'][agent_name][0],
                                        self.token['agents_to_tasks'][agent_name]['goal']]
                    else:
                        closest_task = [self.token['agents_to_tasks'][agent_name]['start'],
                                        self.token['agents_to_tasks'][agent_name]['goal']]
                else:
                    closest_task_name = self.get_closest_task_name(available_tasks, agent_pos)
                    closest_task = available_tasks[closest_task_name]
                moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
                idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
                agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_task[0]}
                env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                  moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep, a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, terraforming_radius=self.terraforming_radius, static_obstacles=self.obstacles, laziness=self.laziness, cell_values=self.cell_values, gamma=self.gamma)
                cbs = CBS(env)
                result_start = self.search(cbs, agent_name, moving_obstacles_agents)
                if not result_start:
                    print("Solution not found to task start for agent", agent_name, " idling at current position...")
                    print("delayed agents:", self.token['delayed_agents'])
                    if len(self.token['delayed_agents']) == 0 and not self.token['prob_exceeded']:
                        print('Instance is not well-formed or a_star_max_iter is too low for this environment.')
                        self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, RECOVERY_RADIUS)
                        # exit(1)
                    #else:
                    #   import sys
                    #    print("\n" + "!"*70)
                    #    print("FATAL ERROR: PLANNING FAILED!")
                    #    print(f"Agent '{agent_name}' could not find a path to TASK START {closest_task[0]}.")
                    #    print(f"Current Agent Pos: {agent_pos}")
                    #    print(f"Possible causes: 'laziness' parameter too low, 'a_star_max_iter' exceeded, or dynamic topological deadlock.")
                    #    print("Halting simulation as requested.")
                    #    print("!"*70 + "\n")
                    #    sys.exit(1)
                else:
                    path_to_task_start = result_start['agents']
                    start_obs_trajectories = result_start['obstacles']
                    print("Solution found to task start for agent", agent_name, " searching solution to task goal...")
                    
                    cost1 = len(path_to_task_start[agent_name])
                    
                    # Use cost - 1 because idle cost is 1
                    moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], cost1 - 1)
                    idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents,
                                                                           cost1 - 1)
                    updated_obs_pos = {}
                    abs_start_time = self.simulation.get_time() - 1 + (cost1 - 1)
                    for i in range(len(self.movable_obstacles_list)):
                        obs_id = f"obs_{i}"
                        if obs_id in start_obs_trajectories and len(start_obs_trajectories[obs_id]) > 0:
                            last_pos = start_obs_trajectories[obs_id][-1]
                            updated_obs_pos[obs_id] = [{'t': 0, 'x': last_pos['x'], 'y': last_pos['y']}]
                        else:
                            path = self.token['obstacles_paths'][obs_id]
                            valid_steps = [step for step in path if step['t'] <= abs_start_time]
                            base_pos = max(valid_steps, key=lambda s: s['t']) if valid_steps else ({'x': 0, 'y': 0} if not path else path[0])
                            rel_path = [{'t': 0, 'x': base_pos['x'], 'y': base_pos['y']}]
                            for step in path:
                                if step['t'] >= abs_start_time:
                                    rel_path.append({'t': step['t'] - abs_start_time, 'x': step['x'], 'y': step['y']})
                            updated_obs_pos[obs_id] = rel_path
                    agent = {'name': agent_name, 'start': closest_task[0], 'goal': closest_task[1]}
                    env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                      moving_obstacles_agents, movable_obstacles=updated_obs_pos, original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep, a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, terraforming_radius=self.terraforming_radius, static_obstacles=self.obstacles, laziness=self.laziness, cell_values=self.cell_values, gamma=self.gamma)
                    cbs = CBS(env)
                    result_goal = self.search(cbs, agent_name, moving_obstacles_agents)
                    if not result_goal:
                        print("Solution not found to task goal for agent", agent_name, " idling at current position...")
                        print("delayed agents:", self.token['delayed_agents'])
                        if len(self.token['delayed_agents']) == 0 and not self.token['prob_exceeded']:
                            print('Instance is not well-formed or a_star_max_iter is too low for this environment.')
                            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, RECOVERY_RADIUS)
                        #else:
                        #    import sys
                        #   print("\n" + "!"*70)
                        #    print("FATAL ERROR: PLANNING FAILED!")
                        #    print(f"Agent '{agent_name}' could not find a path to TASK GOAL {closest_task[1]}.")
                        #    print(f"Current Agent Pos: {closest_task[0]} (Task Start)")
                        #    print(f"Possible causes: 'laziness' parameter too low, 'a_star_max_iter' exceeded, or dynamic topological deadlock.")
                        #    print("Halting simulation as requested.")
                        #    print("!"*70 + "\n")
                        #    sys.exit(1)
                    else:
                        print("Solution found to task goal for agent", agent_name, " doing task...")
                        path_to_task_goal = result_goal['agents']
                        goal_obs_trajectories = result_goal['obstacles']
                        
                        cost2 = len(path_to_task_goal[agent_name])
                        
                        current_sim_time = self.simulation.get_time()
                        base_time = current_sim_time - 1
                        for obs_id, traj in start_obs_trajectories.items():
                            if len(traj) > 0:
                                first_abs_t = base_time + traj[0]['t']
                                self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                            for step in traj:
                                abs_t = base_time + step['t']
                                self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                        for obs_id, traj in goal_obs_trajectories.items():
                            if len(traj) > 0:
                                first_abs_t = base_time + (cost1 - 1) + traj[0]['t']
                                self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                            for step in traj:
                                abs_t = base_time + (cost1 - 1) + step['t']
                                self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                        if agent_name not in self.token['agents_to_tasks']:
                            self.token['tasks'].pop(closest_task_name)
                            task = available_tasks.pop(closest_task_name)
                        else:
                            task = closest_task
                        if agent_name in self.token['delayed_agents_to_reach_task_start']:
                            self.token['delayed_agents_to_reach_task_start'].remove(agent_name)
                        last_step = path_to_task_goal[agent_name][-1]
                        self.update_ends(agent_pos)
                        self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                        self.token['agents_to_tasks'][agent_name] = {'task_name': closest_task_name, 'start': task[0],
                                                                     'goal': task[1], 'predicted_cost': cost1 + cost2}
                        self.token['agents'][agent_name] = []
                        for el in path_to_task_start[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
                        # Don't repeat twice same step
                        self.token['agents'][agent_name] = self.token['agents'][agent_name][:-1]
                        for el in path_to_task_goal[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
                        self.record_path_visits(self.token['agents'][agent_name])
            elif self.check_safe_idle(agent_pos):
                print('No available tasks for agent', agent_name, ' idling at current position...')
            else:
                self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, all_delayed_agents)

        self.check_plan_conflicts()

        # Advance along paths in the token
        if not self.new_recovery:
            for name, path in self.token['agents'].items():
                if len(path) > 1:
                    self.token['agents'][name] = path[1:]
    
    def check_plan_conflicts(self):
        now = self.simulation.get_time()
        paths = {
            name: [tuple(cell) for cell in path]
            for name, path in self.token['agents'].items()
            if path
        }
        names = list(paths)

        def at(path, time):
            return path[min(time, len(path) - 1)]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agent_a, agent_b = names[i], names[j]
                path_a, path_b = paths[agent_a], paths[agent_b]
                for time in range(max(len(path_a), len(path_b))):
                    if at(path_a, time) == at(path_b, time):
                        print(
                            f"[PLAN CONFLICT] time {now}: vertex "
                            f"{agent_a}-{agent_b} at +{time} in {at(path_a, time)}"
                        )
                        break
                    if (
                        at(path_a, time) == at(path_b, time + 1)
                        and at(path_a, time + 1) == at(path_b, time)
                    ):
                        print(
                            f"[PLAN CONFLICT] time {now}: swap "
                            f"{agent_a}-{agent_b} at +{time} between "
                            f"{at(path_a, time)} and {at(path_a, time + 1)}"
                        )
                        break

    def shift_obstacle_trajectories(self, agent_name, current_time):
        for obs_id, path in self.token['obstacles_paths'].items():
            shifted = False
            for step in path:
                # Target future steps belonging to this specific delayed agent
                if step.get('agent') == agent_name and step['t'] >= current_time:
                    step['t'] += 1
                    shifted = True
            
            # Re-sort to maintain strict chronological order if multiple agents interact
            if shifted:
                self.token['obstacles_paths'][obs_id] = sorted(path, key=lambda s: s['t'])
    
    def scrub_future_obstacle_trajectories(self, agent_name, current_time):
       
        for obs_id, path in self.token['obstacles_paths'].items():
            valid_path = []
            for step in path:
                if step.get('agent') == agent_name and step['t'] >= current_time:
                    continue
                valid_path.append(step)
            self.token['obstacles_paths'][obs_id] = valid_path

    