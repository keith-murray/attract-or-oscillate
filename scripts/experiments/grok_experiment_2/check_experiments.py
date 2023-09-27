import os
import json

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment_2"

def main():
    failed_experiments = []
    
    experiment_id = 0  # Starting index for experiments
    for alpha in [0.1, 1.0]:
        for grok in range(1, 14):
            for grok_label in [None, -1, 1]:
                if grok_label == 1 and grok > 7:
                    continue
                
                # Construct the path to the experiment folder
                experiment_folder = os.path.join(base_folder, f"experiment_{experiment_id}")
                
                experiment_failed = False
                # Loop over each task/trial in the experiment
                for task_id in range(192):
                    # Construct the path to the metrics history CSV file
                    task_folder = os.path.join(experiment_folder, f"task_{task_id}")
                    metrics_file = os.path.join(task_folder, "metrics_history.csv")
                    
                    if not os.path.exists(metrics_file):
                        experiment_failed = True
                        break
                
                if experiment_failed:
                    failed_experiments.append(experiment_id)
                    
                experiment_id += 1
    
    # Write the failed experiment IDs to a JSON file
    with open('failed_experiments.json', 'w') as f:
        json.dump(failed_experiments, f)
        
    print(f"Identified {len(failed_experiments)} failed experiments. Details are written to 'failed_experiments.json'")

if __name__ == "__main__":
    main()
