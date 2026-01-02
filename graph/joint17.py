import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# 17-joint skeleton structure (COCO format)
# Joint indices: 0-16
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

num_node = 17
self_link = [(i, i) for i in range(num_node)]

# Bone connections (child, parent) in 1-indexed format
# COCO skeleton structure:
# Head: Nose connects to eyes, eyes to ears
# Torso: Shoulders connect to hips, shoulders connect to each other
# Arms: Shoulder -> Elbow -> Wrist
# Legs: Hip -> Knee -> Ankle
inward_ori_index = [
    # Head connections (1-indexed: joints 1-5)
    (2, 1),   # Left Eye -> Nose
    (3, 1),   # Right Eye -> Nose
    (4, 2),   # Left Ear -> Left Eye
    (5, 3),   # Right Ear -> Right Eye
    
    # Torso connections (1-indexed: joints 6-7, 12-13)
    (7, 6),   # Right Shoulder -> Left Shoulder
    (12, 6),  # Left Hip -> Left Shoulder
    (13, 7),  # Right Hip -> Right Shoulder
    (13, 12), # Right Hip -> Left Hip
    
    # Left arm (1-indexed: joints 6, 8, 10)
    (8, 6),   # Left Elbow -> Left Shoulder
    (10, 8),  # Left Wrist -> Left Elbow
    
    # Right arm (1-indexed: joints 7, 9, 11)
    (9, 7),   # Right Elbow -> Right Shoulder
    (11, 9),  # Right Wrist -> Right Elbow
    
    # Left leg (1-indexed: joints 12, 14, 16)
    (14, 12), # Left Knee -> Left Hip
    (16, 14), # Left Ankle -> Left Knee
    
    # Right leg (1-indexed: joints 13, 15, 17)
    (15, 13), # Right Knee -> Right Hip
    (17, 15), # Right Ankle -> Right Knee
]

# Convert to 0-indexed
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
