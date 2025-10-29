#!/usr/bin/env python3
"""
Display AG and NDVI images from the dataset side by side
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Load dataset metadata
with open('cropnet_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

# Select one sample from each crop type
crop_types = ["Corn", "Soybean", "Winter Wheat", "Cotton"]
samples = []

for crop in crop_types:
    crop_samples = [s for s in data if s['crop_type'] == crop]
    samples.append(random.choice(crop_samples))

# Create figure with 2 rows (AG and NDVI) and 4 columns (crop types)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('CropNet Dataset Images: AG (top) and NDVI (bottom)', fontsize=16)

# Plot each sample
for i, sample in enumerate(samples):
    # Load images
    ag_image = np.load(sample['ag_image_path'])
    ndvi_image = np.load(sample['ndvi_image_path'])
    
    # AG image (top row)
    axes[0, i].imshow(np.clip(ag_image, 0, 1))
    axes[0, i].set_title(f"{sample['crop_type']}\nFIPS: {sample['fips']}, Year: {sample['year']}")
    axes[0, i].axis('off')
    
    # NDVI image (bottom row)
    im = axes[1, i].imshow(ndvi_image, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1, i].set_title(f"Yield: {sample['usda_yield']}")
    axes[1, i].axis('off')

# Add colorbar for NDVI images
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('NDVI Value')

# Add row labels
fig.text(0.08, 0.75, 'AG Images\n(RGB)', ha='center', va='center', fontsize=14, rotation=90)
fig.text(0.08, 0.25, 'NDVI Images\n(Vegetation Index)', ha='center', va='center', fontsize=14, rotation=90)

plt.tight_layout(rect=[0.1, 0, 0.9, 0.95])
plt.savefig('ag_ndvi_comparison.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'ag_ndvi_comparison.png'")

# Show image details
for i, sample in enumerate(samples):
    print(f"\nSample {i+1}: {sample['crop_type']}")
    print(f"  FIPS: {sample['fips']}")
    print(f"  Year: {sample['year']}")
    print(f"  Growth Stage: {sample['growth_stage']:.2f}")
    print(f"  Yield: {sample['usda_yield']}")

