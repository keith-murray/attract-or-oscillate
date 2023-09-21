import os
import json

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment"
script_folder = "/home/gridsan/kmurray/attract-or-oscillate/src"

batch_script_template = """#!/bin/bash

# Initialize and load modules
source /etc/profile
module load anaconda/2023a-tensorflow

# Output task info
echo "My task ID: $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"

# Execute the Python script for this task
python {script_path} $LLSUB_RANK {experiment_folder}
"""

def create_experiment(alpha, grok, grok_label, experiment_id):
    experiment_name = f"experiment_{experiment_id}"
    experiment_folder = os.path.join(base_folder, experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    
    for task_id in range(96):
        task_folder = os.path.join(experiment_folder, f"task_{task_id}")
        os.makedirs(task_folder, exist_ok=True)
        
        params = {
            'seed': task_id,
            'alpha': alpha,
            'min_samples': 50,
            'test_samples': 20,
            'norm_clip': 0.25,
            'batch_size': 256,
            'epochs': 1000,
            'grok': grok,
            'grok_label': grok_label,
        }
        
        json_path = os.path.join(task_folder, "params.json")
        with open(json_path, 'w') as f:
            json.dump(params, f)
    
    batch_script_content = batch_script_template.format(script_path=os.path.join(script_folder, 'pipeline.py'),
                                                        experiment_folder=experiment_folder)
    batch_script_path = os.path.join(experiment_folder, "run.sh")
    with open(batch_script_path, 'w') as f:
        f.write(batch_script_content)


experiment_id = 0  # Starting index for experiments
for alpha in [0.1, 1.0]:
    for grok in [1, 2, 3, 4, 5]:
        for grok_label in [None, -1, 1]:
            create_experiment(alpha, grok, grok_label, experiment_id)
            experiment_id += 1