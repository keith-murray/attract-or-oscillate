import os
import subprocess

base_folder = "/home/gridsan/kmurray/attract-or-oscillate/results/grok_experiment_2"

# Iterate over all experiment folders in base_folder
for experiment_name in os.listdir(base_folder):
    experiment_folder = os.path.join(base_folder, experiment_name)
    
    # Check if the path is a directory, skip if it's a file or a symlink
    if not os.path.isdir(experiment_folder):
        continue
    
    batch_script_path = os.path.join(experiment_folder, "run.sh")
    
    # Change the permissions of run.sh to make it executable
    subprocess.run(["chmod", "u+x", batch_script_path])
    
    # Run the experiment
    subprocess.run(["LLsub", batch_script_path, "[4,48,1]"])
