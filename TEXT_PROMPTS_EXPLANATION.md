# Text Prompts and Contrastive Learning in GAP

## Overview

The GAP (Graph-based Action Prediction) system uses **contrastive learning** with CLIP-style text encoders to improve action recognition. The model learns to align skeleton features with text descriptions of actions.

## How It Works

### 1. **Text Prompt Loading** (`Text_Prompt.py`)

At the start of training, text prompts are loaded:

```python
# In main_multipart_yolo_ucla.py (line 61-62)
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part_ucla()
text_list = text_prompt_openai_random_ucla()
```

**Two types of text prompts are used:**

#### A. `text_dict` (Pre-tokenized, Fixed Prompts)
- **Source**: `text/ucla_pasta_openai_t01.txt`
- **Format**: Each line contains action descriptions separated by semicolons (`;`)
- **Example**: 
  ```
  pick up with one hand;head: tilts slightly forward; hand: reaches down and grasps object; ...
  ```
- **Usage**: Creates 5 different text augmentations (indices 0-4) by combining different parts:
  - `ind=0`: First part only
  - `ind=1`: First 2 parts combined
  - `ind=2`: First part + parts 2-4
  - `ind=3`: First part + part 4
  - `ind=4`: First part + parts 5+
- **Pre-processing**: All prompts are tokenized using CLIP tokenizer and stored as tensors

#### B. `text_list` (Random Selection)
- **Source**: `text/ucla_synonym_openai_t01.txt`
- **Format**: Each line contains synonyms separated by commas (`,`)
- **Example**:
  ```
  pick up with one hand,Lift, heave, hoist, raise, upend, upturn, upheave, ...
  ```
- **Usage**: For each sample in a batch, randomly selects one synonym from the list
- **Tokenization**: Done on-the-fly during training

### 2. **CLIP Text Encoder Setup**

During model initialization (`Processor.load_model()`):

```python
# Load CLIP model (e.g., 'ViT-B/32')
model_, preprocess = clip.load(name, clip_device)
del model_.visual  # Remove vision encoder, keep only text encoder
model_text = TextCLIP(model_)  # Wrapper for text encoding
self.model_text_dict[name] = model_text
```

- **TextCLIP**: Wrapper that uses CLIP's text encoder (`encode_text()`)
- **Purpose**: Converts text tokens to embeddings that match the skeleton feature space

### 3. **Training Loop - Text Selection and Encoding**

During each training step (`Processor.train()`):

```python
for ind in range(num_text_aug):  # num_text_aug = 5
    if ind > 0:
        # Use pre-tokenized prompts from text_dict
        text_id = np.ones(len(label), dtype=np.int8) * ind
        text_tensors = [text_dict[j][i,:] for i,j in zip(label, text_id)]
        texts = torch.stack(text_tensors)
    else:
        # Use random synonym selection from text_list
        texts = []
        for i in range(len(label)):
            text_len = len(text_list[label[i]])
            text_id = np.random.randint(text_len, size=1)
            text_item = text_list[label[i]][text_id.item()]
            texts.append(text_item)
        texts = torch.cat(texts)
    
    # Encode text to embeddings
    text_embedding = self.model_text_dict['ViT-B/32'](texts).float()
```

**Text Selection Strategy:**
- **ind=0**: Random synonym selection (data augmentation)
- **ind=1-4**: Fixed pre-tokenized prompts (different combinations of action parts)

### 4. **Contrastive Learning**

The model uses a **dual-stream architecture**:

1. **Skeleton Stream**: `self.model(data)` → skeleton features
2. **Text Stream**: `self.model_text_dict['ViT-B/32'](texts)` → text embeddings

**Loss Computation:**

```python
# Get skeleton features from model
output, feature_dict, logit_scale, part_feature_list = self.model(data)

# For each text augmentation (ind=0 to 4):
text_embedding = self.model_text_dict['ViT-B/32'](texts).float()

# Create logits (similarity scores)
logits_per_image, logits_per_text = create_logits(
    feature_dict['ViT-B/32'],  # Skeleton features
    text_embedding,              # Text embeddings
    logit_scale[:,0].mean()      # Temperature scaling
)

# Ground truth: positive pairs have same label
label_g = gen_label(label)  # Creates one-hot matrix for same-label pairs

# Compute contrastive loss (KL divergence)
loss_imgs = self.loss(logits_per_image, ground_truth)
loss_texts = self.loss(logits_per_text, ground_truth)
loss_te = (loss_imgs + loss_texts) / 2

# Total loss = classification loss + contrastive loss
loss_ce = self.loss_ce(output, label)  # Cross-entropy
loss = loss_ce + loss_alpha * sum(loss_te_list) / len(loss_te_list)
```

### 5. **Key Functions**

#### `create_logits(x1, x2, logit_scale)` (`tools.py`)
- Normalizes features to unit vectors
- Computes cosine similarity: `logits = logit_scale * (x1 @ x2.T)`
- Returns similarity matrix between skeleton features and text embeddings

#### `gen_label(labels)` (`tools.py`)
- Creates ground truth matrix for contrastive learning
- `gt[i,k] = 1` if `labels[i] == labels[k]` (same action class)
- Used to identify positive pairs in the batch

#### `KLLoss()` (`KLLoss.py`)
- KL divergence loss for contrastive learning
- Measures how well predicted similarities match ground truth pairs

## File Structure

```
text/
├── ucla_synonym_openai_t01.txt    # Synonyms for random selection (text_list)
├── ucla_pasta_openai_t01.txt      # Part-based descriptions (text_dict)
└── ntu120_label_map.txt           # Action class names (for NTU dataset)

Text_Prompt.py                      # Functions to load and process text prompts
```

## Data Flow

```
Text Files
    ↓
Text_Prompt.py (tokenize, organize)
    ↓
text_dict (5 pre-tokenized variants)
text_list (random synonyms)
    ↓
Training Loop
    ↓
Select text based on augmentation index
    ↓
CLIP Text Encoder (ViT-B/32)
    ↓
Text Embeddings
    ↓
Contrastive Loss (align with skeleton features)
```

## Benefits

1. **Multi-modal Learning**: Aligns visual (skeleton) and textual (action descriptions) representations
2. **Data Augmentation**: Multiple text descriptions per action improve robustness
3. **Semantic Understanding**: Model learns semantic relationships between actions
4. **Transfer Learning**: CLIP's pre-trained text encoder provides rich semantic features

## Example: UCLA Action "pick up with one hand"

**Synonyms** (`text_list`):
- "pick up with one hand"
- "Lift"
- "heave"
- "hoist"
- etc.

**Part-based** (`text_dict[0]`):
- "pick up with one hand"

**Part-based** (`text_dict[1]`):
- "pick up with one hand;head: tilts slightly forward"

**Part-based** (`text_dict[2]`):
- "pick up with one hand;head: tilts slightly forward; hand: reaches down and grasps object; arm: extends and lifts object"

During training, the model learns that skeleton features for "picking up" should be similar to embeddings of all these text descriptions.
