#!/usr/bin/env python3
"""
Detailed analysis of a single AG and NDVI image pair
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Load dataset metadata
with open('cropnet_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

# Select a corn sample
corn_samples = [s for s in data if s['crop_type'] == 'Corn']
sample = random.choice(corn_samples)

print(f"Selected sample:")
print(f"  Crop: {sample['crop_type']}")
print(f"  FIPS: {sample['fips']}")
print(f"  Year: {sample['year']}")
print(f"  Growth Stage: {sample['growth_stage']:.2f}")
print(f"  Weather Quality: {sample['weather_quality']:.2f}")
print(f"  Yield: {sample['usda_yield']} bushels/acre")

# Load images
ag_image = np.load(sample['ag_image_path'])
ndvi_image = np.load(sample['ndvi_image_path'])

print(f"\nAG Image: Shape {ag_image.shape}, Range [{np.min(ag_image):.2f}, {np.max(ag_image):.2f}]")
print(f"NDVI Image: Shape {ndvi_image.shape}, Range [{np.min(ndvi_image):.2f}, {np.max(ndvi_image):.2f}]")

# Create detailed visualization
fig = plt.figure(figsize=(15, 12))
fig.suptitle(f"Detailed Analysis: {sample['crop_type']} Field - FIPS {sample['fips']}, Year {sample['year']}", fontsize=16)

# 1. AG Image (RGB)
ax1 = plt.subplot(3, 3, 1)
ax1.imshow(np.clip(ag_image, 0, 1))
ax1.set_title('AG Image (RGB)')
ax1.axis('off')

# 2-4. Individual RGB channels
channel_names = ['Red', 'Green', 'Blue']
cmaps = ['Reds', 'Greens', 'Blues']
for i in range(3):
    ax = plt.subplot(3, 3, i+2)
    im = ax.imshow(ag_image[:,:,i], cmap=cmaps[i])
    ax.set_title(f'{channel_names[i]} Channel')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# 5. NDVI Image
ax5 = plt.subplot(3, 3, 4)
im5 = ax5.imshow(ndvi_image, cmap='RdYlGn', vmin=-1, vmax=1)
ax5.set_title('NDVI Image')
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046)

# 6. NDVI Histogram
ax6 = plt.subplot(3, 3, 5)
ax6.hist(ndvi_image.flatten(), bins=50, color='green', alpha=0.7)
ax6.set_title('NDVI Distribution')
ax6.set_xlabel('NDVI Value')
ax6.set_ylabel('Frequency')
ax6.grid(alpha=0.3)

# 7. RGB Histograms
ax7 = plt.subplot(3, 3, 6)
for i, color in enumerate(['red', 'green', 'blue']):
    ax7.hist(ag_image[:,:,i].flatten(), bins=50, color=color, alpha=0.5, label=channel_names[i])
ax7.set_title('RGB Channels Distribution')
ax7.set_xlabel('Pixel Value')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Field Patterns - Horizontal Profile
ax8 = plt.subplot(3, 3, 7)
row = 112  # Middle row
ax8.plot(ag_image[row,:,0], 'r-', alpha=0.7, label='Red')
ax8.plot(ag_image[row,:,1], 'g-', alpha=0.7, label='Green')
ax8.plot(ag_image[row,:,2], 'b-', alpha=0.7, label='Blue')
ax8.plot(ndvi_image[row,:], 'k-', alpha=0.7, label='NDVI')
ax8.set_title('Horizontal Profile (Middle Row)')
ax8.set_xlabel('Pixel Position')
ax8.set_ylabel('Value')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Field Patterns - Vertical Profile
ax9 = plt.subplot(3, 3, 8)
col = 112  # Middle column
ax9.plot(ag_image[:,col,0], 'r-', alpha=0.7, label='Red')
ax9.plot(ag_image[:,col,1], 'g-', alpha=0.7, label='Green')
ax9.plot(ag_image[:,col,2], 'b-', alpha=0.7, label='Blue')
ax9.plot(ndvi_image[:,col], 'k-', alpha=0.7, label='NDVI')
ax9.set_title('Vertical Profile (Middle Column)')
ax9.set_xlabel('Pixel Position')
ax9.set_ylabel('Value')
ax9.legend()
ax9.grid(alpha=0.3)

# 10. Sample Information
ax10 = plt.subplot(3, 3, 9)
ax10.axis('off')
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
    f"NDVI Image: {ndvi_image.shape}\n\n"
    f"AG Range: [{np.min(ag_image):.2f}, {np.max(ag_image):.2f}]\n"
    f"NDVI Range: [{np.min(ndvi_image):.2f}, {np.max(ndvi_image):.2f}]"
)
ax10.text(0.05, 0.95, info_text, va='top', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('detailed_image_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved detailed visualization to 'detailed_image_analysis.png'")

