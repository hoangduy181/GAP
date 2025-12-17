# Functions Used in `main_multipart_ucla.py`

This document provides a comprehensive overview of all functions used in `main_multipart_ucla.py`, including local functions, imported functions, and class methods.

## Table of Contents

1. [Local Functions](#local-functions)
2. [Processor Class Methods](#processor-class-methods)
3. [Imported Functions](#imported-functions)
4. [External Dependencies](#external-dependencies)

---

## Local Functions

### `init_seed(seed)`
**Location:** Lines 52-59  
**Purpose:** Initialize random seeds for reproducibility across PyTorch, NumPy, and Python's random module.

**Parameters:**
- `seed` (int): Random seed value

**Behavior:**
- Sets CUDA random seed for all GPUs
- Sets PyTorch random seed
- Sets NumPy random seed
- Sets Python random seed
- Configures cuDNN for deterministic (False) or benchmark (True) mode

**Usage:**
```python
init_seed(1)  # Initialize with seed 1
```

---

### `import_class(import_str)`
**Location:** Lines 61-67  
**Purpose:** Dynamically import a class from a module string.

**Parameters:**
- `import_str` (str): String representation of class path (e.g., "model.ctrgcn.Model")

**Returns:**
- Class object

**Raises:**
- `ImportError`: If class cannot be found

**Usage:**
```python
Model = import_class("model.ctrgcn.Model")
```

---

### `str2bool(v)`
**Location:** Lines 69-75  
**Purpose:** Convert string to boolean value for argparse.

**Parameters:**
- `v` (str): String value to convert

**Returns:**
- `True` for: 'yes', 'true', 't', 'y', '1'
- `False` for: 'no', 'false', 'f', 'n', '0'

**Raises:**
- `argparse.ArgumentTypeError`: For unsupported values

**Usage:**
```python
value = str2bool("yes")  # Returns True
```

---

### `get_parser()`
**Location:** Lines 78-225  
**Purpose:** Create and configure argparse parser with all command-line arguments.

**Returns:**
- `argparse.ArgumentParser`: Configured parser object

**Key Arguments Defined:**
- **Work Directory:** `--work-dir` (default: './work_dir/temp')
- **Configuration:** `--config` (default: './config/nturgbd-cross-view/test_bone.yaml')
- **Phase:** `--phase` (default: 'train', options: 'train' or 'test')
- **Model:** `--model`, `--model-args`, `--weights`, `--ignore-weights`
- **Training:** `--base-lr`, `--batch-size`, `--num-epoch`, `--optimizer`, `--device`
- **Data:** `--feeder`, `--train-feeder-args`, `--test-feeder-args`
- **Logging:** `--log-interval`, `--save-interval`, `--eval-interval`, `--print-log`
- **Learning Rate:** `--step`, `--lr-decay-rate`, `--warm_up_epoch`
- **Loss:** `--loss-alpha`, `--te-lr-ratio`

**Usage:**
```python
parser = get_parser()
args = parser.parse_args()
```

---

## Processor Class Methods

The `Processor` class (lines 228-664) handles the main training and evaluation logic for skeleton-based action recognition.

### `Processor.__init__(arg)`
**Location:** Lines 233-287  
**Purpose:** Initialize the Processor with configuration arguments.

**Parameters:**
- `arg`: Parsed arguments object from argparse

**Initialization Steps:**
1. Saves arguments to config file
2. Sets up TensorBoard writers for training/validation
3. Loads model and text encoders
4. Loads optimizer and data loaders
5. Configures multi-GPU support if needed

**Key Attributes Created:**
- `self.arg`: Configuration arguments
- `self.model`: Main skeleton action recognition model
- `self.model_text_dict`: Dictionary of text encoder models (CLIP-based)
- `self.optimizer`: Optimizer (SGD or Adam)
- `self.data_loader`: Dictionary with 'train' and 'test' data loaders
- `self.train_writer`, `self.val_writer`: TensorBoard summary writers
- `self.loss_ce`: Cross-entropy loss function
- `self.loss`: KL divergence loss function

---

### `Processor.load_data()`
**Location:** Lines 289-306  
**Purpose:** Load and configure data loaders for training and testing.

**Creates:**
- `self.data_loader['train']`: Training DataLoader (if phase == 'train')
- `self.data_loader['test']`: Testing DataLoader

**DataLoader Configuration:**
- Uses `import_class()` to dynamically load feeder class
- Training: shuffle=True, drop_last=True
- Testing: shuffle=False, drop_last=False
- Both use `init_seed` as `worker_init_fn` for reproducibility

---

### `Processor.load_model()`
**Location:** Lines 308-358  
**Purpose:** Load the main model, text encoders, and optionally pre-trained weights.

**Process:**
1. Determines output device (first GPU in list or single GPU)
2. Dynamically imports model class using `import_class()`
3. Instantiates main skeleton model
4. Creates CrossEntropyLoss and KLLoss instances
5. Loads CLIP-based text encoders for each head
6. Optionally loads pre-trained weights from file (.pkl or .pt)

**Text Encoder Setup:**
- Loads CLIP models specified in `self.arg.model_args['head']`
- Removes visual component (only uses text encoder)
- Wraps in `TextCLIP` class
- Stores in `self.model_text_dict`

**Weight Loading:**
- Supports `.pkl` (pickle) and `.pt` (PyTorch) formats
- Handles DataParallel module prefix removal
- Filters ignored weights
- Gracefully handles missing keys

---

### `Processor.load_optimizer()`
**Location:** Lines 361-379  
**Purpose:** Initialize optimizer with separate learning rates for model and text encoders.

**Supported Optimizers:**
- **SGD:** With momentum (0.9), nesterov option, weight decay
  - Model parameters: `base_lr`
  - Text encoder parameters: `base_lr * te_lr_ratio`
- **Adam:** With weight decay
  - Only model parameters (text encoders use same LR)

**Usage:**
```python
# SGD with different LR for text encoders
optimizer = optim.SGD([
    {'params': model.parameters(), 'lr': 0.001},
    {'params': text_encoders.parameters(), 'lr': 0.001 * te_lr_ratio}
], ...)
```

---

### `Processor.save_arg()`
**Location:** Lines 381-388  
**Purpose:** Save configuration arguments to YAML file in work directory.

**Output:**
- `{work_dir}/config.yaml`: Contains command-line arguments and defaults

**Format:**
- YAML format with command-line invocation as comment

---

### `Processor.adjust_learning_rate(epoch)`
**Location:** Lines 390-402  
**Purpose:** Adjust learning rate based on warm-up and decay schedule.

**Parameters:**
- `epoch` (int): Current epoch number

**Learning Rate Schedule:**
1. **Warm-up phase** (epoch < warm_up_epoch):
   - Linear increase: `lr = base_lr * (epoch + 1) / warm_up_epoch`
2. **Decay phase** (epoch >= warm_up_epoch):
   - Step decay: `lr = base_lr * (lr_decay_rate ** num_steps_passed)`
   - Steps defined by `--step` argument (default: [20, 40, 60])

**Returns:**
- `lr` (float): Updated learning rate

---

### `Processor.print_log(str, print_time=True)`
**Location:** Lines 408-415  
**Purpose:** Print log message with optional timestamp and save to file.

**Parameters:**
- `str` (str): Message to log
- `print_time` (bool): Whether to prepend timestamp

**Output:**
- Console: Prints message
- File: Appends to `{work_dir}/log.txt` (if `arg.print_log` is True)

**Format:**
- `[ Mon Jan 01 12:00:00 2024 ] message`

---

### `Processor.record_time()`
**Location:** Lines 417-419  
**Purpose:** Record current time for performance measurement.

**Returns:**
- Current timestamp (float)

---

### `Processor.split_time()`
**Location:** Lines 421-424  
**Purpose:** Calculate time elapsed since last `record_time()` call.

**Returns:**
- Time difference in seconds (float)

**Usage:**
```python
self.record_time()
# ... do work ...
elapsed = self.split_time()  # Get elapsed time
```

---

### `Processor.train(epoch, save_model=False)`
**Location:** Lines 426-529  
**Purpose:** Train the model for one epoch.

**Parameters:**
- `epoch` (int): Current epoch number
- `save_model` (bool): Whether to save model checkpoint

**Training Process:**
1. Set model to training mode
2. Adjust learning rate
3. Iterate through training batches:
   - Forward pass with mixed precision (AMP)
   - Compute multi-part text-image contrastive loss
   - Compute classification loss
   - Backward pass with gradient scaling
   - Update optimizer
4. Log metrics to TensorBoard (loss, accuracy, learning rate)
5. Save model checkpoint if `save_model=True`

**Loss Components:**
- **Classification Loss:** Cross-entropy on model output
- **Text-Image Loss:** KL divergence between image features and text embeddings
  - Multiple text augmentations (`num_text_aug`)
  - Part-based features for multi-part learning
  - Weighted by `loss_alpha`

**Text Augmentation Strategy:**
- `ind == 0`: Random text selection from `text_list`
- `ind > 0`: Fixed text from `text_dict` based on augmentation index

**Metrics Logged:**
- Training accuracy
- Training loss
- Learning rate
- Time consumption breakdown

---

### `Processor.eval(epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None)`
**Location:** Lines 531-610  
**Purpose:** Evaluate the model on test/validation set.

**Parameters:**
- `epoch` (int): Current epoch number
- `save_score` (bool): Whether to save prediction scores
- `loader_name` (list): List of data loader names to evaluate
- `wrong_file` (str, optional): Path to save incorrectly predicted samples
- `result_file` (str, optional): Path to save all predictions

**Evaluation Process:**
1. Set model to evaluation mode
2. Iterate through batches without gradient computation
3. Compute predictions and loss
4. Calculate accuracy metrics (Top-K)
5. Generate confusion matrix
6. Save per-class accuracy to CSV
7. Optionally save scores and prediction files

**Outputs:**
- Console: Accuracy, loss, Top-K metrics
- TensorBoard: Validation loss and accuracy
- Files:
  - `epoch{epoch}_{loader}_score.pkl`: Prediction scores
  - `epoch{epoch}_{loader}_each_class_acc.csv`: Per-class accuracy and confusion matrix
  - `{wrong_file}`: Incorrect predictions (if provided)
  - `{result_file}`: All predictions (if provided)

**Metrics:**
- Top-K accuracy (configurable via `--show-topk`)
- Per-class accuracy
- Confusion matrix

---

### `Processor.start()`
**Location:** Lines 612-664  
**Purpose:** Main entry point for training or testing.

**Training Mode (`phase == 'train'`):**
1. Print model parameters count
2. Training loop:
   - Train for one epoch
   - Evaluate on test set
   - Save model at intervals
3. After training:
   - Load best model (highest accuracy)
   - Final evaluation with detailed outputs
   - Print training summary

**Testing Mode (`phase == 'test'`):**
1. Load specified weights
2. Run evaluation
3. Save results

**Training Summary Includes:**
- Best accuracy
- Best epoch
- Model name
- Total parameters
- Training hyperparameters (weight decay, LR, batch size, seed)

---

## Imported Functions

### From `tools.py`

#### `gen_label(labels)`
**Location:** Line 454  
**Purpose:** Generate ground truth matrix for contrastive learning.

**Parameters:**
- `labels` (array-like): Class labels for batch

**Returns:**
- `gt` (numpy.ndarray): Square matrix where `gt[i,j] = 1` if `labels[i] == labels[j]`

**Usage:**
```python
label_g = gen_label(label)  # Creates positive pairs matrix
```

---

#### `create_logits(x1, x2, logit_scale)`
**Location:** Lines 478, 482  
**Purpose:** Create logits from normalized feature vectors using cosine similarity.

**Parameters:**
- `x1` (torch.Tensor): First set of features (e.g., image features)
- `x2` (torch.Tensor): Second set of features (e.g., text embeddings)
- `logit_scale` (float): Temperature scaling factor

**Returns:**
- `logits_per_x1` (torch.Tensor): Logits for x1 against x2
- `logits_per_x2` (torch.Tensor): Logits for x2 against x1

**Process:**
1. L2-normalize both feature sets
2. Compute cosine similarity: `x1 @ x2.t()`
3. Scale by `logit_scale` (temperature)

**Usage:**
```python
logits_img, logits_text = create_logits(image_features, text_embeddings, logit_scale)
```

---

### From `Text_Prompt.py`

#### `text_prompt_openai_pasta_pool_4part_ucla()`
**Location:** Line 38  
**Purpose:** Load pre-generated text prompts for NW-UCLA dataset with 4-part augmentation.

**Returns:**
- `classes`: List of class names
- `num_text_aug`: Number of text augmentations
- `text_dict`: Dictionary mapping (class_idx, aug_idx) to text embeddings

**Usage:**
```python
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part_ucla()
```

---

#### `text_prompt_openai_random_ucla()`
**Location:** Line 40  
**Purpose:** Load random text prompts for NW-UCLA dataset.

**Returns:**
- `text_list`: List of lists, where `text_list[i]` contains text prompts for class `i`

**Usage:**
```python
text_list = text_prompt_openai_random_ucla()
# Access: text_list[class_idx][prompt_idx]
```

---

### From `KLLoss.py`

#### `KLLoss`
**Location:** Line 35, Lines 317, 487-488  
**Purpose:** KL divergence loss for contrastive learning.

**Usage:**
```python
loss_fn = KLLoss().cuda(device)
loss_imgs = loss_fn(logits_per_image, ground_truth)
loss_texts = loss_fn(logits_per_text, ground_truth)
```

**Purpose:**
- Computes KL divergence between predicted logits and ground truth distribution
- Used for text-image contrastive learning

---

## External Dependencies

### PyTorch Functions
- `torch.cuda.amp.GradScaler()`: Mixed precision training
- `torch.cuda.amp.autocast()`: Automatic mixed precision context
- `torch.nn.DataParallel`: Multi-GPU parallelization
- `torch.utils.data.DataLoader`: Data loading
- `torch.save()`, `torch.load()`: Model checkpointing

### CLIP Functions
- `clip.load(name, device)`: Load pre-trained CLIP model

### NumPy Functions
- `np.ones()`, `np.random.randint()`, `np.concatenate()`, `np.mean()`, `np.diag()`, `np.sum()`, `np.arange()`

### Other Libraries
- `yaml.safe_load()`, `yaml.dump()`: Configuration file handling
- `SummaryWriter` (TensorBoard): Logging and visualization
- `tqdm`: Progress bars
- `sklearn.metrics.confusion_matrix`: Confusion matrix computation
- `csv.writer`: CSV file writing
- `pickle.dump()`, `pickle.load()`: Score serialization

---

## Global Variables

### `device`
**Location:** Line 43  
**Type:** str  
**Value:** `"cuda"` if CUDA available, else `"cpu"`

### `scaler`
**Location:** Line 45  
**Type:** `torch.cuda.amp.GradScaler`  
**Purpose:** Gradient scaler for mixed precision training

### `classes`, `num_text_aug`, `text_dict`
**Location:** Line 38  
**Purpose:** Text prompt data loaded at module import

### `text_list`
**Location:** Line 40  
**Purpose:** Random text prompts loaded at module import

---

## Main Execution Flow

```python
if __name__ == '__main__':
    # 1. Parse command-line arguments
    parser = get_parser()
    p = parser.parse_args()
    
    # 2. Load configuration from YAML file
    if p.config is not None:
        default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
    
    # 3. Parse final arguments
    arg = parser.parse_args()
    
    # 4. Initialize random seed
    init_seed(arg.seed)
    
    # 5. Create processor
    processor = Processor(arg)
    
    # 6. Start training/testing
    processor.start()
```

---

## Notes

- The script uses mixed precision training (AMP) for faster training and reduced memory usage
- Multi-GPU support is handled via `nn.DataParallel`
- Text encoders use CLIP models with visual components removed
- The model supports multi-part learning with separate text augmentations for each part
- Evaluation includes per-class accuracy and confusion matrix generation
- Model checkpoints are saved with epoch and global step in filename

