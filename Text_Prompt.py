"""
Text Prompt Module for Contrastive Learning with CLIP

This module provides text prompt generation and augmentation strategies for action recognition
using contrastive learning. It loads action class labels and generates various text descriptions
that are used to train CLIP-based text encoders alongside skeleton-based action recognition models.

The text prompts help align skeleton features with text descriptions of actions, enabling
the model to learn a joint embedding space where similar actions (both in skeleton and text)
are close together.

Key Concepts:
- Text Augmentation: Multiple text descriptions per action class to improve robustness
- CLIP Tokenization: Converting text strings to tokenized tensors for CLIP text encoder
- Contrastive Learning: Learning by contrasting positive pairs (matching skeleton-text) 
  against negative pairs (non-matching skeleton-text)
"""

import torch
import clip


# Load action class labels from NTU120 dataset
# Each line contains the text description of an action class (e.g., "drink water", "throw")
label_text_map = []

with open('text/ntu120_label_map.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())



# Load synonym-based text augmentations for NTU dataset
# Format: Each line contains comma-separated synonyms for one action class
# Example: "drink, consume, sip, gulp" for the "drink water" action
# Used for random synonym selection during training
paste_text_map0 = []

with open('text/synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        paste_text_map0.append(temp_list)
        

# Load sentence-based text augmentations for NTU dataset
# Format: Each line contains period-separated sentences describing one action
# Example: "A person is drinking water. The person takes a sip. ..."
# Note: Padded to at least 4 sentences per action
paste_text_map1 = []

with open('text/sentence_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split('.')
        while len(temp_list) < 4:
            temp_list.append(" ")
        paste_text_map1.append(temp_list)


# Load PASTA (Part-Aware Semantic Text Augmentation) descriptions for NTU dataset
# Format: Each line contains semicolon-separated parts describing different aspects of an action
# Example: "drink water; using hand; near mouth; standing; indoors"
# This format allows creating different text combinations by selecting different parts
# Used for multi-part text augmentation (4-part or 5-part combinations)
paste_text_map2 = []

with open('text/pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        paste_text_map2.append(temp_list)

# Load synonym-based text augmentations for UCLA dataset
# Similar to paste_text_map0 but for UCLA action classes
ucla_paste_text_map0 = []

with open('text/ucla_synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        ucla_paste_text_map0.append(temp_list)


# Load PASTA descriptions for UCLA dataset
# Similar to paste_text_map2 but for UCLA action classes
ucla_paste_text_map1 = []

with open('text/ucla_pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ucla_paste_text_map1.append(temp_list)




def text_prompt():
    """
    Generate basic text prompts with template-based augmentation for NTU dataset.
    
    This function creates multiple text descriptions for each action class by applying
    different templates (e.g., "a photo of action {drink water}", "Human action of {drink water}").
    This augmentation helps the model learn robust text-skeleton alignments.
    
    Returns:
        classes: Concatenated tokenized text for all actions and all templates
        num_text_aug: Number of text augmentation templates (16)
        text_dict: Dictionary mapping template index to tokenized text tensor
                  Shape: {template_idx: (num_classes, max_seq_len)}
    """
    # Define 16 different text templates for action description
    # Each template will be filled with action class names from label_text_map
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    # For each template, create tokenized text for all action classes
    # Example: template "a photo of action {}" becomes "a photo of action drink water" for each class
    for ii, txt in enumerate(text_aug):
        # Tokenize each action class with the current template
        # clip.tokenize() converts text to token IDs that CLIP can process
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])

    # Concatenate all templates for all classes into a single tensor
    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


def text_prompt_openai_random():
    """
    Generate random synonym-based text prompts for NTU dataset.
    
    This function loads synonyms for each action class and tokenizes them.
    During training, one synonym is randomly selected per action to create
    diverse text-skeleton pairs, improving generalization.
    
    Returns:
        total_list: List of lists, where each inner list contains tokenized synonyms
                   for one action class
                   Shape: (num_classes, num_synonyms_per_class)
    """
    print("Use text prompt openai synonym random")

    total_list = []
    # For each action class, tokenize all its synonyms
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list

def text_prompt_openai_random_bert():
    """
    Generate random synonym-based text prompts for BERT (not CLIP tokenization).
    
    Similar to text_prompt_openai_random() but returns raw text strings instead of
    CLIP tokenized tensors. Used when BERT is the text encoder instead of CLIP.
    
    Returns:
        total_list: List of lists, where each inner list contains raw synonym strings
                   for one action class (not tokenized)
    """
    print("Use text prompt openai synonym random bert")
    
    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(item)  # Keep as raw string, not tokenized
        total_list.append(temp_list)
    return total_list



def text_prompt_openai_pasta_pool_4part():
    """
    Generate PASTA (Part-Aware Semantic Text Augmentation) prompts for NTU dataset.
    
    PASTA creates multiple text descriptions by combining different parts of action descriptions.
    Each action has multiple parts (e.g., "drink water", "using hand", "near mouth", etc.)
    separated by semicolons. This function creates 5 different combinations:
    
    - Part 0: Use only the first part (main action)
    - Part 1: Combine parts 0-1
    - Part 2: Combine part 0 with parts 2-3
    - Part 3: Combine part 0 with part 4
    - Part 4: Combine part 0 with parts 5+
    
    This multi-part approach allows the model to learn from different levels of action detail,
    improving robustness and enabling part-aware learning.
    
    Returns:
        classes: Concatenated tokenized text for all actions and all part combinations
        num_text_aug: Number of part combinations (5)
        text_dict: Dictionary mapping part combination index to tokenized text tensor
    """
    print("Use text prompt openai pasta pool")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            # Part 0: Use only the first part (main action description)
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in paste_text_map2])
        elif ii == 1:
            # Part 1: Combine first two parts
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in paste_text_map2])
        elif ii == 2:
            # Part 2: Combine part 0 with parts 2-3
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in paste_text_map2])
        elif ii == 3:
            # Part 3: Combine part 0 with part 4
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in paste_text_map2])
        else:
            # Part 4: Combine part 0 with parts 5 and beyond
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in paste_text_map2])


    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict

def text_prompt_openai_pasta_pool_4part_bert():
    """
    Generate PASTA prompts for BERT (not CLIP tokenization).
    
    Similar to text_prompt_openai_pasta_pool_4part() but returns raw text strings
    instead of CLIP tokenized tensors. Used when BERT is the text encoder.
    
    Returns:
        num_text_aug: Number of part combinations (5)
        text_dict: Dictionary mapping part combination index to list of raw text strings
    """
    print("Use text prompt openai pasta pool bert")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            input_list = [pasta_list[ii] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 1:
            input_list = [','.join(pasta_list[0:2]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 2:
            input_list = [pasta_list[0] +','.join(pasta_list[2:4]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 3:
            input_list = [pasta_list[0] +','+ pasta_list[4] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        else:
            input_list = [pasta_list[0]+','+','.join(pasta_list[5:]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list

    
    return num_text_aug, text_dict



def text_prompt_openai_random_ucla():
    """
    Generate random synonym-based text prompts for UCLA dataset.
    
    Similar to text_prompt_openai_random() but uses UCLA-specific synonyms.
    UCLA dataset has 10 action classes, so this function works with those classes.
    
    Returns:
        total_list: List of lists, where each inner list contains tokenized synonyms
                   for one UCLA action class
    """
    print("Use text prompt openai synonym random UCLA")

    total_list = []
    for pasta_list in ucla_paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list


def text_prompt_openai_pasta_pool_4part_ucla():
    """
    Generate PASTA prompts for UCLA dataset.
    
    Similar to text_prompt_openai_pasta_pool_4part() but uses UCLA-specific PASTA descriptions.
    UCLA dataset has 10 action classes, and this function creates 5 different part combinations
    for each class, following the same strategy as the NTU version.
    
    This is the main function used for UCLA training in main_multipart_yolo_ucla.py.
    
    Returns:
        classes: Concatenated tokenized text for all UCLA actions and all part combinations
        num_text_aug: Number of part combinations (5)
        text_dict: Dictionary mapping part combination index to tokenized text tensor
    """
    print("Use text prompt openai pasta pool ucla")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            # Part 0: Use only the first part (main action description)
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ucla_paste_text_map1])
        elif ii == 1:
            # Part 1: Combine first two parts
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 2:
            # Part 2: Combine part 0 with parts 2-3
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 3:
            # Part 3: Combine part 0 with part 4
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in ucla_paste_text_map1])
        else:
            # Part 4: Combine part 0 with parts 5 and beyond
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in ucla_paste_text_map1])



    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict

