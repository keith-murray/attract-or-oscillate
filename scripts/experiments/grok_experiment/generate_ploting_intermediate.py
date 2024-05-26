import os
import pandas as pd
import json

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment_2"

# Initialize a dictionary to store the count of near-perfect grokking accuracy
perfect_grok_counts = {
    "0.1": {"None": {}, "-1": {}, "1": {}},
    "1.0": {"None": {}, "-1": {}, "1": {}},
}

# Load the list of failed experiments from failed_experiments.json
failed_experiments_file = 'failed_experiments.json'
if os.path.exists(failed_experiments_file):
    with open(failed_experiments_file, 'r') as f:
        failed_experiments = json.load(f)
else:
    failed_experiments = []

experiment_id = 0  # Starting index for experiments
for alpha in [0.1, 1.0]:
    str_alpha = str(alpha)
    for grok in range(1, 27):
        for grok_label in [None, -1, 1]:
            if grok_label == 1 and grok > 8:
                continue
            elif grok_label == -1 and grok > 17:
                continue
            
            str_grok_label = "None" if grok_label is None else str(grok_label)
            
            if experiment_id in failed_experiments:
                # Skip counting for this experiment as it is in the list of failed experiments
                experiment_id += 1
                continue
            
            # Initialize counter for the experiment
            perfect_grok_count = 0

            # Construct the path to the experiment folder
            experiment_folder = os.path.join(base_folder, f"experiment_{experiment_id}")

            # Loop over each task/trial in the experiment
            for task_id in range(192):
                # Construct the path to the metrics history CSV file
                task_folder = os.path.join(experiment_folder, f"task_{task_id}")
                metrics_file = os.path.join(task_folder, "metrics_history.csv")
                
                if os.path.exists(metrics_file):
                    # Load the metrics history using pandas
                    df = pd.read_csv(metrics_file)
                    # Check if there are any rows where grokking accuracy is 0.95 or more
                    if (df['grok_accuracy'] >= 0.95).any():
                        perfect_grok_count += 1

            # Store the count in the dictionary
            perfect_grok_counts[str_alpha][str_grok_label][str(grok)] = perfect_grok_count
            experiment_id += 1

# Save the dictionary as a JSON file
with open('near_perfect_grok_counts.json', 'w') as f:
    json.dump(perfect_grok_counts, f)

print("near_perfect_grok_counts.json has been created.")