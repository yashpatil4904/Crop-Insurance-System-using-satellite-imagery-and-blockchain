#!/usr/bin/env python3
"""
Visualize AG and NDVI image samples from the CropNet dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Load dataset metadata
print("Loading dataset metadata...")
with open('cropnet_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

# Select samples - one from each crop type
crop_types = ["Corn", "Soybean", "Winter Wheat", "Cotton"]
samples = []

for crop in crop_types:
    crop_samples = [s for s in data if s['crop_type'] == crop]
    samples.append(random.choice(crop_samples))

print(f"Selected {len(samples)} samples (one per crop type)")

# Create figure for visualization
plt.figure(figsize=(16, 10))

# Plot each sample
for i, sample in enumerate(samples):
    # Load images
    ag_image = np.load(sample['ag_image_path'])
    ndvi_image = np.load(sample['ndvi_image_path'])
    
    # AG image - RGB
    plt.subplot(2, 4, i+1)
    plt.imshow(np.clip(ag_image, 0, 1))
    plt.title(f"{sample['crop_type']} - AG Image\nFIPS: {sample['fips']}, Year: {sample['year']}")
    plt.axis('off')
    
    # NDVI image - single channel
    plt.subplot(2, 4, i+5)
    plt.imshow(ndvi_image, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title(f"{sample['crop_type']} - NDVI Image\nYield: {sample['usda_yield']}")
    plt.axis('off')
    
    # Print info
    print(f"\nSample {i+1}: {sample['crop_type']}")
    print(f"  FIPS: {sample['fips']}")
    print(f"  Year: {sample['year']}")
    print(f"  Growth Stage: {sample['growth_stage']:.2f}")
    print(f"  Yield: {sample['usda_yield']}")
    print(f"  AG Image: {ag_image.shape}, Range: [{np.min(ag_image):.2f}, {np.max(ag_image):.2f}]")
    print(f"  NDVI Image: {ndvi_image.shape}, Range: [{np.min(ndvi_image):.2f}, {np.max(ndvi_image):.2f}]")

plt.tight_layout()
plt.savefig('cropnet_sample_images.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to 'cropnet_sample_images.png'")

# Create detailed single sample visualization
sample = samples[0]  # Use the first sample (Corn)

plt.figure(figsize=(18, 6))

# AG image and its channels
ag_image = np.load(sample['ag_image_path'])

# Full RGB image
plt.subplot(2, 4, 1)
plt.imshow(np.clip(ag_image, 0, 1))
plt.title('AG Image (RGB)')
plt.axis('off')

# Individual RGB channels
channel_names = ['Red', 'Green', 'Blue']
cmaps = ['Reds', 'Greens', 'Blues']
for i in range(3):
    plt.subplot(2, 4, i+2)
    plt.imshow(ag_image[:,:,i], cmap=cmaps[i])
    plt.title(f'{channel_names[i]} Channel')
    plt.axis('off')
    plt.colorbar(fraction=0.046)

# NDVI image
ndvi_image = np.load(sample['ndvi_image_path'])
plt.subplot(2, 4, 5)
plt.imshow(ndvi_image, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title('NDVI Image')
plt.axis('off')
plt.colorbar(fraction=0.046)

# NDVI histogram
plt.subplot(2, 4, 6)
plt.hist(ndvi_image.flatten(), bins=50, color='green', alpha=0.7)
plt.title('NDVI Distribution')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# RGB histogram
plt.subplot(2, 4, 7)
for i, color in enumerate(['red', 'green', 'blue']):
    plt.hist(ag_image[:,:,i].flatten(), bins=50, color=color, alpha=0.5, label=channel_names[i])
plt.title('RGB Channels Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

# Sample info
plt.subplot(2, 4, 8)
plt.axis('off')
info_text = (
    f"SAMPLE INFORMATION\n"
    f"------------------\n"
    f"Crop: {sample['crop_type']}\n"
    f"FIPS: {sample['fips']}\n"
    f"Year: {sample['year']}\n"
    f"Growth Stage: {sample['growth_stage']:.2f}\n"
    f"Weather Quality: {sample['weather_quality']:.2f}\n"
    f"Yield: {sample['usda_yield']} bushels/acre\n\n"
    f"AG Image: {ag_image.shape}\n"
    f"NDVI Image: {ndvi_image.shape}"
)
plt.text(0.1, 0.5, info_text, fontsize=12, va='center')

plt.tight_layout()
plt.savefig('cropnet_detailed_sample.png', dpi=150, bbox_inches='tight')
print("Saved detailed visualization to 'cropnet_detailed_sample.png'")
