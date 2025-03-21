import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_scalars_from_event(logdir, tag='loss'):
    """Load scalar values (e.g., training loss) from TensorBoard event files."""
    event_files = glob.glob(os.path.join(logdir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {logdir}")
        return [], []

    # Load the first event file
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    print(ea.Tags()['scalars'])
    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' not found in {logdir}")
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_losses(logdirs, labels, tag='loss', save_path='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    
    for logdir, label in zip(logdirs, labels):
        steps, values = load_scalars_from_event(logdir, tag)
        if steps:
            plt.plot(steps, values, label=label)
    
    plt.xlabel('Step')
    plt.ylabel(tag)
    plt.title(f'Training {tag}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Example usage:
if __name__ == "__main__":
    logdirs = [
        "tensorboard/bmae_pretrain_layer_6_time_1",  # Replace with your actual paths
        "tensorboard/bmae_pretrain_layer_6_time_5",
        "tensorboard/bmae_pretrain",
        "tensorboard/bmae_pretrain_layer_6_time_10"

    ]
    labels = [
        "time = 1",
        "time = 5",
        "time = 3",
        "time = 10"
    ]
    tag = "train_loss"  # Replace with your actual scalar tag
    plot_losses(logdirs, labels, tag, save_path="comparison_loss_plot.png")