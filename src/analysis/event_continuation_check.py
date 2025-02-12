import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

def get_first_and_last_step(event_file):
    """Extract the first and last step from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()

    min_step, max_step = float('inf'), float('-inf')
    
    for tag in ea.Tags()['scalars']:
        steps = [event.step for event in ea.Scalars(tag)]
        if steps:
            min_step = min(min_step, min(steps))
            max_step = max(max_step, max(steps))

    return min_step if min_step != float('inf') else None, max_step if max_step != float('-inf') else None

# File paths
event_file1 = "/home/psaha03/scratch/outputs/kits/central/events/events.out.tfevents.1739209250.cdr386.int.cedar.computecanada.ca.1658787.0"
event_file2 = "/home/psaha03/scratch/outputs/kits/central/events/events.out.tfevents.1739259035.cdr257.int.cedar.computecanada.ca.615045.0"

# Get first and last steps
min_step1, max_step1 = get_first_and_last_step(event_file1)
min_step2, max_step2 = get_first_and_last_step(event_file2)

# Check continuity
if max_step1 is not None and min_step2 is not None:
    if min_step2 > max_step1:
        print(f"The second log continues from the first (steps {max_step1} → {min_step2}).")
    else:
        print(f"The second log does NOT continue from the first (overlapping or reset at step {min_step2}).")
else:
    print("Could not determine step ranges for one or both files.")
