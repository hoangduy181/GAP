# Training Log Visualization

This script parses the training log file and creates two graphs:
1. **Loss Graph**: Training loss vs Evaluation loss over epochs
2. **Accuracy Graph**: Training accuracy vs Evaluation Top-1 and Top-5 accuracy over epochs

## Requirements

```bash
pip install matplotlib numpy
```

## Usage

```bash
python parse_training_log.py
```

The script will:
- Parse `training/ucla/test_run_1/gan_log.txt`
- Generate two PNG files:
  - `training/ucla/test_run_1/loss_graph.png`
  - `training/ucla/test_run_1/accuracy_graph.png`

## Output

The graphs will show:
- **Loss Graph**: Blue line for training loss, red line for evaluation loss
- **Accuracy Graph**: Blue line for training accuracy, red line for eval Top-1, green line for eval Top-5

All graphs are saved as high-resolution PNG files (300 DPI) suitable for presentations and papers.

