import json
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D 
import RoothPath

def draw_map():
    
    config_path = os.path.join(RoothPath.get_root(), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Errore: config.json non trovato!")
        return

    map_file = os.path.join(RoothPath.get_root(), config['input_path'], config['input_name'])
    print(f"Sto leggendo la mappa: {map_file}")

   
    with open(map_file, 'r') as f:
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

    
    plt.title(f"Map Preview: {config['input_name']}", fontsize=16, pad=15)

    output_filename = "map_preview.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nMappa generata con successo! Salvata come: {output_filename}")
    
    
    plt.show()

if __name__ == '__main__':
    draw_map()