#!/usr/bin/env python3
"""
Parse training log files and create CSV files and visualization graphs for loss and accuracy.
"""

import re
import csv
import sys
import os
import math

# Try to import numpy, fall back to math.nan if not available
try:
    import numpy as np
    HAS_NUMPY = True
    nan = np.nan
    isnan = np.isnan
except ImportError:
    HAS_NUMPY = False
    nan = float('nan')
    isnan = math.isnan

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Graphs will not be generated.")
    print("Install with: pip install matplotlib")

def parse_log_file(log_path):
    """Parse the training log file and extract metrics."""
    train_data = {}  # epoch -> {loss, acc}
    eval_data = {}   # epoch -> {loss, top1, top5}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for training epoch
        train_match = re.search(r'Training epoch: (\d+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            # Look for training loss and accuracy in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                loss_match = re.search(r'Mean training loss: ([\d.]+)', lines[j])
                acc_match = re.search(r'Mean training acc: ([\d.]+)%', lines[j])
                if loss_match:
                    loss_str = loss_match.group(1).rstrip('.')  # Remove trailing period
                    train_data[epoch] = {'loss': float(loss_str)}
                if acc_match:
                    if epoch not in train_data:
                        train_data[epoch] = {}
                    acc_str = acc_match.group(1).rstrip('.')  # Remove trailing period
                    train_data[epoch]['acc'] = float(acc_str)
        
        # Check for eval epoch
        eval_match = re.search(r'Eval epoch: (\d+)', line)
        if eval_match:
            epoch = int(eval_match.group(1))
            # Look for eval metrics in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                loss_match = re.search(r'Mean test loss of \d+ batches: ([\d.]+)', lines[j])
                top1_match = re.search(r'Top1: ([\d.]+)%', lines[j])
                top5_match = re.search(r'Top5: ([\d.]+)%', lines[j])
                if loss_match:
                    if epoch not in eval_data:
                        eval_data[epoch] = {}
                    loss_str = loss_match.group(1).rstrip('.')  # Remove trailing period
                    eval_data[epoch]['loss'] = float(loss_str)
                if top1_match:
                    if epoch not in eval_data:
                        eval_data[epoch] = {}
                    top1_str = top1_match.group(1).rstrip('.')  # Remove trailing period
                    eval_data[epoch]['top1'] = float(top1_str)
                if top5_match:
                    if epoch not in eval_data:
                        eval_data[epoch] = {}
                    top5_str = top5_match.group(1).rstrip('.')  # Remove trailing period
                    eval_data[epoch]['top5'] = float(top5_str)
        
        i += 1
    
    return train_data, eval_data

def merge_data(main_train, main_eval, cont_train, cont_eval):
    """Merge continuation data into main data (continuation overrides for overlapping epochs)."""
    # Start with main data
    merged_train = main_train.copy()
    merged_eval = main_eval.copy()
    
    # Override with continuation data (for overlapping epochs)
    merged_train.update(cont_train)
    merged_eval.update(cont_eval)
    
    return merged_train, merged_eval

def create_csv(epochs, train_losses, train_accs, eval_losses, eval_top1, eval_top5, csv_path):
    """Create CSV file with all metrics."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Top1 (%)', 'Val Top5 (%)'])
        
        for epoch, t_loss, t_acc, v_loss, v_top1, v_top5 in zip(
            epochs, train_losses, train_accs, eval_losses, eval_top1, eval_top5
        ):
            writer.writerow([
                epoch,
                '' if isnan(t_loss) else f'{t_loss:.6f}',
                '' if isnan(t_acc) else f'{t_acc:.2f}',
                '' if isnan(v_loss) else f'{v_loss:.6f}',
                '' if isnan(v_top1) else f'{v_top1:.2f}',
                '' if isnan(v_top5) else f'{v_top5:.2f}'
            ])
    
    print(f"CSV file saved to: {csv_path}")

def create_loss_graph(epochs, train_losses, eval_losses, output_path):
    """Create loss graph for train and eval."""
    plt.figure(figsize=(12, 6))
    
    # Filter out NaN values for plotting
    train_epochs = [e for e, l in zip(epochs, train_losses) if not isnan(l)]
    train_vals = [l for l in train_losses if not isnan(l)]
    
    eval_epochs = [e for e, l in zip(epochs, eval_losses) if not isnan(l)]
    eval_vals = [l for l in eval_losses if not isnan(l)]
    
    plt.plot(train_epochs, train_vals, 'b-', label='Train Loss', linewidth=2, alpha=0.8, marker='o', markersize=3)
    plt.plot(eval_epochs, eval_vals, 'r-', label='Val Loss', linewidth=2, alpha=0.8, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss graph saved to: {output_path}")
    plt.close()

def create_accuracy_graph(epochs, train_accs, eval_top1, eval_top5, output_path):
    """Create accuracy graph for train and eval."""
    plt.figure(figsize=(12, 6))
    
    # Filter out NaN values for plotting
    train_epochs = [e for e, a in zip(epochs, train_accs) if not isnan(a)]
    train_vals = [a for a in train_accs if not isnan(a)]
    
    eval_top1_epochs = [e for e, a in zip(epochs, eval_top1) if not isnan(a)]
    eval_top1_vals = [a for a in eval_top1 if not isnan(a)]
    
    eval_top5_epochs = [e for e, a in zip(epochs, eval_top5) if not isnan(a)]
    eval_top5_vals = [a for a in eval_top5 if not isnan(a)]
    
    plt.plot(train_epochs, train_vals, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8, marker='o', markersize=3)
    plt.plot(eval_top1_epochs, eval_top1_vals, 'r-', label='Val Top-1 Accuracy', linewidth=2, alpha=0.8, marker='s', markersize=3)
    plt.plot(eval_top5_epochs, eval_top5_vals, 'g-', label='Val Top-5 Accuracy', linewidth=2, alpha=0.8, marker='^', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])  # Accuracy is in percentage
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy graph saved to: {output_path}")
    plt.close()

def main():
    # Define log file paths
    main_log_path = 'gap_log.txt'
    cont_log_path = 'gap_run_cont.txt'
    
    # Parse main log file
    print(f"Parsing main log file: {main_log_path}")
    main_train_data, main_eval_data = parse_log_file(main_log_path)
    print(f"  Found {len(main_train_data)} training epochs and {len(main_eval_data)} eval epochs")
    
    # Parse continuation log file
    print(f"Parsing continuation log file: {cont_log_path}")
    cont_train_data, cont_eval_data = parse_log_file(cont_log_path)
    print(f"  Found {len(cont_train_data)} training epochs and {len(cont_eval_data)} eval epochs")
    
    # Merge data (continuation overrides main for overlapping epochs)
    train_data, eval_data = merge_data(main_train_data, main_eval_data, cont_train_data, cont_eval_data)
    print(f"Merged data: {len(train_data)} training epochs and {len(eval_data)} eval epochs")
    
    # Get all epochs and sort them
    all_epochs = sorted(set(list(train_data.keys()) + list(eval_data.keys())))
    
    # Build aligned lists
    epochs = []
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_top1 = []
    eval_top5 = []
    
    for epoch in all_epochs:
        epochs.append(epoch)
        
        # Training data
        if epoch in train_data:
            train_losses.append(train_data[epoch].get('loss', nan))
            train_accs.append(train_data[epoch].get('acc', nan))
        else:
            train_losses.append(nan)
            train_accs.append(nan)
        
        # Eval data
        if epoch in eval_data:
            eval_losses.append(eval_data[epoch].get('loss', nan))
            eval_top1.append(eval_data[epoch].get('top1', nan))
            eval_top5.append(eval_data[epoch].get('top5', nan))
        else:
            eval_losses.append(nan)
            eval_top1.append(nan)
            eval_top5.append(nan)
    
    print(f"\nTotal epochs processed: {len(epochs)}")
    print(f"  Epoch range: {min(epochs)} to {max(epochs)}")
    
    # Create output directory if it doesn't exist
    output_dir = '.'
    
    # Create CSV file
    csv_path = os.path.join(output_dir, 'training_metrics.csv')
    create_csv(epochs, train_losses, train_accs, eval_losses, eval_top1, eval_top5, csv_path)
    
    # Create graphs if matplotlib is available
    if HAS_MATPLOTLIB:
        loss_graph_path = os.path.join(output_dir, 'loss_graph.png')
        accuracy_graph_path = os.path.join(output_dir, 'accuracy_graph.png')
        
        create_loss_graph(epochs, train_losses, eval_losses, loss_graph_path)
        create_accuracy_graph(epochs, train_accs, eval_top1, eval_top5, accuracy_graph_path)
        
        print("\nProcessing completed successfully!")
        print(f"  - CSV file: {csv_path}")
        print(f"  - Loss graph: {loss_graph_path}")
        print(f"  - Accuracy graph: {accuracy_graph_path}")
    else:
        print("\nCSV file created successfully!")
        print(f"  - CSV file: {csv_path}")
        print("  (Graphs not generated - matplotlib not installed)")

if __name__ == '__main__':
    main()
