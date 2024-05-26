import os
import subprocess

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment_2"

def all_trials_completed(experiment_folder):
    """Check if all trial folders in the given experiment folder have a metrics_history.csv file."""
    # Get the list of all tasks/trials in the experiment folder
    for task_name in os.listdir(experiment_folder):
        # Only consider folders with "task" in the name
        if "task" not in task_name:
            continue
            
        task_folder = os.path.join(experiment_folder, task_name)
        
        # Check if the path is a directory, skip if it's a file or a symlink
        if not os.path.isdir(task_folder):
            continue
        
        metrics_file = os.path.join(task_folder, "metrics_history.csv")
        if not os.path.exists(metrics_file):
            # If any trial is missing a metrics_history.csv file, return False
            return False
    
    return True

# Iterate over all experiment folders in base_folder
for experiment_name in os.listdir(base_folder):
    experiment_folder = os.path.join(base_folder, experiment_name)
    
    # Check if the path is a directory, skip if it's a file or a symlink
    if not os.path.isdir(experiment_folder):
        continue
    
    if all_trials_completed(experiment_folder):
        # Skip this experiment as all trial folders have a metrics_history.csv file
        continue
    
    batch_script_path = os.path.join(experiment_folder, "run.sh")
    
    # Change the permissions of run.sh to make it executable
    subprocess.run(["chmod", "u+x", batch_script_path])
    
    # Run the experiment
    subprocess.run(["LLsub", batch_script_path, "[4,48,1]"])
