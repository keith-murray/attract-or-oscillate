import os
import pandas as pd
import json

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment_2"

# Initialize a dictionary to store the count of perfect grokking accuracy
perfect_grok_counts = {
    "0.1": {"None": {}, "-1": {}, "1": {}},
    "1.0": {"None": {}, "-1": {}, "1": {}},
}

experiment_id = 0  # Starting index for experiments
for alpha in [0.1, 1.0]:
    str_alpha = str(alpha)
    for grok in range(1, 14):
        for grok_label in [None, -1, 1]:
            str_grok_label = "None" if grok_label is None else str(grok_label)
            if grok_label == 1 and grok > 7:
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
                    # Check if there are any rows where grokking accuracy is 1.0
                    if (df['grok_accuracy'] >= 0.95).any():
                        perfect_grok_count += 1

            # Store the count in the dictionary
            perfect_grok_counts[str_alpha][str_grok_label][str(grok)] = perfect_grok_count
            experiment_id += 1

# Save the dictionary as a JSON file
with open('near_perfect_grok_counts.json', 'w') as f:
    json.dump(perfect_grok_counts, f)

print("perfect_grok_counts.json has been created.")