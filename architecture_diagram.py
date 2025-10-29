#!/usr/bin/env python3
"""
Generate an architecture diagram for the CropNet model
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as PathEffects

def add_box(ax, x, y, width, height, label, color='skyblue', alpha=0.7):
    """Add a box with a label"""
    box = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, 
                   edgecolor='black', linewidth=1)
    ax.add_patch(box)
    
    # Add label
    text = ax.text(x + width/2, y + height/2, label, 
                  ha='center', va='center', fontsize=9, fontweight='bold')
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
    
    return box

def add_arrow(ax, x1, y1, x2, y2, color='black'):
    """Add an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', color=color, 
                           linewidth=1.5, mutation_scale=15)
    ax.add_patch(arrow)
    return arrow

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Title
ax.text(50, 57, 'CropNet: Two-Branch Architecture for Crop Yield Prediction', 
        ha='center', fontsize=16, fontweight='bold')

# Input data boxes
ag_box = add_box(ax, 5, 45, 15, 8, 'AG Image\n(224x224x3)', color='lightgreen')
ndvi_box = add_box(ax, 5, 35, 15, 8, 'NDVI Image\n(224x224)', color='lightgreen')
weather_box = add_box(ax, 5, 25, 15, 8, 'Daily Weather\n(183 days x 9 features)', color='lightblue')
monthly_box = add_box(ax, 5, 15, 15, 8, 'Monthly Weather\n(6 months)', color='lightblue')
metadata_box = add_box(ax, 5, 5, 15, 8, 'Metadata\n(FIPS, crop, year)', color='lightyellow')

# CNN branches
ag_cnn = add_box(ax, 30, 45, 15, 8, 'AG CNN\nResNet18', color='orange')
ndvi_cnn = add_box(ax, 30, 35, 15, 8, 'NDVI CNN\nResNet18', color='orange')

# Weather processors
weather_lstm = add_box(ax, 30, 25, 15, 8, 'Weather LSTM\nBidirectional', color='lightcoral')
monthly_mlp = add_box(ax, 30, 15, 15, 8, 'Monthly MLP', color='lightcoral')
metadata_mlp = add_box(ax, 30, 5, 15, 8, 'Metadata\nEmbeddings + MLP', color='lightcoral')

# Feature fusion
fusion_box = add_box(ax, 55, 25, 15, 10, 'Feature\nFusion\n(Concatenation)', color='mediumpurple')

# Final layers
regression_box = add_box(ax, 80, 25, 15, 10, 'Regression\nLayers\n(MLP)', color='tomato')

# Output
output_box = add_box(ax, 80, 5, 15, 8, 'Yield\nPrediction', color='gold')

# Connect boxes with arrows
add_arrow(ax, 20, 49, 30, 49)  # AG to AG CNN
add_arrow(ax, 20, 39, 30, 39)  # NDVI to NDVI CNN
add_arrow(ax, 20, 29, 30, 29)  # Weather to Weather LSTM
add_arrow(ax, 20, 19, 30, 19)  # Monthly to Monthly MLP
add_arrow(ax, 20, 9, 30, 9)    # Metadata to Metadata MLP

# Connect to fusion
add_arrow(ax, 45, 49, 55, 30)  # AG CNN to Fusion
add_arrow(ax, 45, 39, 55, 30)  # NDVI CNN to Fusion
add_arrow(ax, 45, 29, 55, 30)  # Weather LSTM to Fusion
add_arrow(ax, 45, 19, 55, 30)  # Monthly MLP to Fusion
add_arrow(ax, 45, 9, 55, 30)   # Metadata MLP to Fusion

# Connect fusion to regression
add_arrow(ax, 70, 30, 80, 30)

# Connect regression to output
add_arrow(ax, 87.5, 25, 87.5, 13)

# Add feature dimensions
ax.text(22, 51, "512", fontsize=8)
ax.text(22, 41, "512", fontsize=8)
ax.text(22, 31, "256", fontsize=8)
ax.text(22, 21, "64", fontsize=8)
ax.text(22, 11, "64", fontsize=8)
ax.text(72, 32, "1408", fontsize=8)
ax.text(87.5, 20, "1", fontsize=8)

# Add model components labels
ax.text(5, 54, "Input Data", fontsize=12, fontweight='bold')
ax.text(30, 54, "Feature Extractors", fontsize=12, fontweight='bold')
ax.text(55, 54, "Feature Integration", fontsize=12, fontweight='bold')
ax.text(80, 54, "Prediction", fontsize=12, fontweight='bold')

# Save the figure
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
print("Architecture diagram saved as 'architecture_diagram.png'")
plt.close()

