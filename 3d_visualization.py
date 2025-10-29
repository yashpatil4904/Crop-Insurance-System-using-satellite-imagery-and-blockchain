#!/usr/bin/env python3
"""
3D visualization of AG and NDVI images
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import random

# Load dataset metadata
with open('cropnet_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

# Select a sample with high NDVI values (healthy vegetation)
samples = [s for s in data if s['crop_type'] == 'Corn' and s['growth_stage'] > 0.7]
sample = random.choice(samples)

print(f"Selected sample:")
print(f"  Crop: {sample['crop_type']}")
print(f"  FIPS: {sample['fips']}")
print(f"  Year: {sample['year']}")
print(f"  Growth Stage: {sample['growth_stage']:.2f}")
print(f"  Yield: {sample['usda_yield']} bushels/acre")

# Load images
ag_image = np.load(sample['ag_image_path'])
ndvi_image = np.load(sample['ndvi_image_path'])

# Create figure
fig = plt.figure(figsize=(18, 10))
fig.suptitle(f"{sample['crop_type']} Field - FIPS {sample['fips']}, Year {sample['year']}", fontsize=16)

# 1. AG Image (RGB)
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(np.clip(ag_image, 0, 1))
ax1.set_title('AG Image (RGB)')
ax1.axis('off')

# 2. NDVI Image
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(ndvi_image, cmap='RdYlGn', vmin=-1, vmax=1)
ax2.set_title('NDVI Image')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046)

# 3. Green Channel (vegetation)
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(ag_image[:,:,1], cmap='Greens')
ax3.set_title('Green Channel')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, fraction=0.046)

# 4. 3D Surface of NDVI
ax4 = plt.subplot(2, 3, 4, projection='3d')
# Downsample for performance
step = 4
x = np.arange(0, ndvi_image.shape[1], step)
y = np.arange(0, ndvi_image.shape[0], step)
X, Y = np.meshgrid(x, y)
Z = ndvi_image[::step, ::step]

# Plot the surface
surf = ax4.plot_surface(X, Y, Z, cmap='RdYlGn', linewidth=0, antialiased=False)
ax4.set_title('3D NDVI Surface')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('NDVI Value')
ax4.set_zlim(-0.2, 1.0)
plt.colorbar(surf, ax=ax4, shrink=0.5, aspect=5)

# 5. Crop Row Profile (horizontal slice)
ax5 = plt.subplot(2, 3, 5)
row = 112  # Middle row
ax5.plot(ndvi_image[row,:], 'g-', linewidth=2)
ax5.set_title('NDVI Horizontal Profile (Middle Row)')
ax5.set_xlabel('Pixel Position')
ax5.set_ylabel('NDVI Value')
ax5.grid(alpha=0.3)
ax5.set_ylim(-0.2, 1.0)

# 6. Field Information
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = (
    f"FIELD INFORMATION\n"
    f"------------------\n"
    f"Crop: {sample['crop_type']}\n"
    f"FIPS: {sample['fips']}\n"
    f"Year: {sample['year']}\n"
    f"Growth Stage: {sample['growth_stage']:.2f}\n"
    f"Weather Quality: {sample['weather_quality']:.2f}\n"
    f"Yield: {sample['usda_yield']} bushels/acre\n\n"
    f"NDVI Statistics:\n"
    f"  Min: {np.min(ndvi_image):.2f}\n"
    f"  Max: {np.max(ndvi_image):.2f}\n"
    f"  Mean: {np.mean(ndvi_image):.2f}\n"
    f"  Std Dev: {np.std(ndvi_image):.2f}\n\n"
    f"AG Image Statistics:\n"
    f"  Red Channel Mean: {np.mean(ag_image[:,:,0]):.2f}\n"
    f"  Green Channel Mean: {np.mean(ag_image[:,:,1]):.2f}\n"
    f"  Blue Channel Mean: {np.mean(ag_image[:,:,2]):.2f}"
)
ax6.text(0.05, 0.95, info_text, va='top', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3d_ndvi_visualization.png', dpi=150, bbox_inches='tight')
print("\nSaved 3D visualization to '3d_ndvi_visualization.png'")

