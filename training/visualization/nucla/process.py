#!/usr/bin/env python3
"""
Parse training log file and create visualization graphs for loss and accuracy.
"""

import re
import numpy as np
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

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
            train_losses.append(train_data[epoch].get('loss', np.nan))
            train_accs.append(train_data[epoch].get('acc', np.nan))
        else:
            train_losses.append(np.nan)
            train_accs.append(np.nan)
        
        # Eval data
        if epoch in eval_data:
            eval_losses.append(eval_data[epoch].get('loss', np.nan))
            eval_top1.append(eval_data[epoch].get('top1', np.nan))
            eval_top5.append(eval_data[epoch].get('top5', np.nan))
        else:
            eval_losses.append(np.nan)
            eval_top1.append(np.nan)
            eval_top5.append(np.nan)
    
    return epochs, train_losses, train_accs, eval_losses, eval_top1, eval_top5

def create_loss_graph(epochs, train_losses, eval_losses, output_path):
    """Create loss graph for train and eval."""
    plt.figure(figsize=(12, 6))
    
    # Filter out NaN values for plotting
    train_epochs = [e for e, l in zip(epochs, train_losses) if not np.isnan(l)]
    train_vals = [l for l in train_losses if not np.isnan(l)]
    
    eval_epochs = [e for e, l in zip(epochs, eval_losses) if not np.isnan(l)]
    eval_vals = [l for l in eval_losses if not np.isnan(l)]
    
    plt.plot(train_epochs, train_vals, 'b-', label='Train Loss', linewidth=2, alpha=0.8, marker='o', markersize=3)
    plt.plot(eval_epochs, eval_vals, 'r-', label='Eval Loss', linewidth=2, alpha=0.8, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
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
    train_epochs = [e for e, a in zip(epochs, train_accs) if not np.isnan(a)]
    train_vals = [a for a in train_accs if not np.isnan(a)]
    
    eval_top1_epochs = [e for e, a in zip(epochs, eval_top1) if not np.isnan(a)]
    eval_top1_vals = [a for a in eval_top1 if not np.isnan(a)]
    
    eval_top5_epochs = [e for e, a in zip(epochs, eval_top5) if not np.isnan(a)]
    eval_top5_vals = [a for a in eval_top5 if not np.isnan(a)]
    
    plt.plot(train_epochs, train_vals, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8, marker='o', markersize=3)
    plt.plot(eval_top1_epochs, eval_top1_vals, 'r-', label='Eval Top-1 Accuracy', linewidth=2, alpha=0.8, marker='s', markersize=3)
    plt.plot(eval_top5_epochs, eval_top5_vals, 'g-', label='Eval Top-5 Accuracy', linewidth=2, alpha=0.8, marker='^', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Evaluation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])  # Accuracy is in percentage
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy graph saved to: {output_path}")
    plt.close()

def main():
    log_path = '../../ucla/test_run_1/gan_log.txt'
    
    print(f"Parsing log file: {log_path}")
    epochs, train_losses, train_accs, eval_losses, eval_top1, eval_top5 = parse_log_file(log_path)
    
    print(f"Found {len(epochs)} epochs")
    print(f"Train losses: {len(train_losses)}")
    print(f"Eval losses: {len(eval_losses)}")
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'training/ucla/test_run_1'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graphs
    loss_graph_path = os.path.join(output_dir, 'loss_graph.png')
    accuracy_graph_path = os.path.join(output_dir, 'accuracy_graph.png')
    
    create_loss_graph(epochs, train_losses, eval_losses, loss_graph_path)
    create_accuracy_graph(epochs, train_accs, eval_top1, eval_top5, accuracy_graph_path)
    
    print("\nGraphs created successfully!")
    print(f"  - Loss graph: {loss_graph_path}")
    print(f"  - Accuracy graph: {accuracy_graph_path}")

if __name__ == '__main__':
    main()

