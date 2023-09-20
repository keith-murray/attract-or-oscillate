#!/bin/bash

# Initialize and load modules
source /etc/profile
module load anaconda/2023a-tensorflow

# Output task info
echo "My task ID: $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"

# Execute the Python script for this task
python /home/gridsan/kmurray/attract-or-oscillate/src/pipeline.py $LLSUB_RANK /home/gridsan/kmurray/attract-or-oscillate/results/experiment_6
