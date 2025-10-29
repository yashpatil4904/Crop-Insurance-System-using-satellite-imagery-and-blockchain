"""
Visualize CropNet Satellite Images
Shows how Sentinel-2 images look and their structure for model training
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

print("="*70)
print("CROPNET SATELLITE IMAGE STRUCTURE FOR MODEL TRAINING")
print("="*70)

# Load the record structure
with open('cropnet_record_example.json', 'r') as f:
    record = json.load(f)

print("\n1. SENTINEL-2 IMAGE SPECIFICATIONS:")
print(f"   - Resolution: 224x224 pixels")
print(f"   - Spatial coverage: 9km x 9km per image")
print(f"   - Revisit frequency: Every 14 days")
print(f"   - Growing season: ~26 images per year (April-October)")

# Create synthetic examples to show what the images look like
print("\n2. CREATING SYNTHETIC SATELLITE IMAGES (for visualization)...")

# Create sample AG (Agriculture RGB) image - simulating crop fields
np.random.seed(42)
ag_image = np.zeros((224, 224, 3), dtype=np.float32)

# Simulate agricultural patterns
rows, cols = 224, 224
for i in range(rows):
    for j in range(cols):
        # Create field patterns with green vegetation
        field_pattern = np.sin(i * 0.05) * np.cos(j * 0.05) * 0.3 + 0.5
        # Simulate growing crops with seasonal variation
        crop_greenness = np.random.uniform(0.4, 0.9) * field_pattern
        
        ag_image[i, j] = [
            crop_greenness * 0.3,      # Red channel (less for green crops)
            crop_greenness * 0.8,      # Green channel (dominant for vegetation)
            crop_greenness * 0.2       # Blue channel
        ]

# Ensure values are in [0, 1]
ag_image = np.clip(ag_image, 0, 1)

# Create sample NDVI image - vegetation index
ndvi_image = np.random.uniform(0.2, 0.9, (224, 224))
# Add some structure to NDVI (cropped areas)
for i in range(0, 224, 40):
    for j in range(0, 224, 40):
        # Field boundaries
        ndvi_image[i:i+3, j:j+40] = 0.1  # Dark lines (field boundaries)
        ndvi_image[i:i+40, j:j+3] = 0.1

# Clip NDVI to valid range [-1, 1], then normalize for display
ndvi_display = (ndvi_image - ndvi_image.min()) / (ndvi_image.max() - ndvi_image.min())

print("\n3. VISUALIZATION:")
print("   Creating figure with both image types...")

# Create visualization
fig = plt.figure(figsize=(15, 8))

# AG Image - RGB
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(ag_image)
ax1.set_title('Sentinel-2 Agriculture (AG) Image\nRGB, 224x224x3', fontsize=12, fontweight='bold')
ax1.axis('off')

# NDVI Image
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(ndvi_display, cmap='RdYlGn', vmin=0, vmax=1)
ax2.set_title('Sentinel-2 NDVI Image\nVegetation Index, 224x224', fontsize=12, fontweight='bold')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Vegetation Index')

# Show individual RGB channels
ax3 = plt.subplot(2, 3, 3)
ax3.imshow(ag_image[:, :, 0], cmap='Reds', vmin=0, vmax=1)
ax3.set_title('Red Channel\n(Shows soil, drought)', fontsize=10)
ax3.axis('off')

ax4 = plt.subplot(2, 3, 4)
ax4.imshow(ag_image[:, :, 1], cmap='Greens', vmin=0, vmax=1)
ax4.set_title('Green Channel\n(Shows vegetation)', fontsize=10)
ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
ax5.imshow(ag_image[:, :, 2], cmap='Blues', vmin=0, vmax=1)
ax5.set_title('Blue Channel\n(Shows water, shadows)', fontsize=10)
ax5.axis('off')

# Temporal sequence (multiple images over growing season)
ax6 = plt.subplot(2, 3, 6)
# Simulate 5 images over time
for idx in range(5):
    time_ndvi = ndvi_image * (0.5 + idx * 0.1)  # Growing over time
    time_ndvi_display = (time_ndvi - time_ndvi.min()) / (time_ndvi.max() - time_ndvi.min())
    ax6.plot([idx], [np.mean(time_ndvi_display)], 'o-', markersize=10, label=f'Day {idx*14}')
    ax6.set_title('NDVI Over Growing Season\n(Every 14 days)', fontsize=10)
    ax6.set_xlabel('Time (bi-weekly)')
    ax6.set_ylabel('Avg NDVI')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

plt.suptitle('CropNet Satellite Images: Structure for Model Training', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('cropnet_satellite_images_structure.png', dpi=150, bbox_inches='tight')
print("   [OK] Saved visualization to 'cropnet_satellite_images_structure.png'")
plt.show()

# Tl summary of image dimensions for ML
print("\n" + "="*70)
print("IMAGE DIMENSIONS FOR MODEL INPUT:")
print("="*70)
print("\n1. Single Image:")
print("   AG Image shape:", ag_image.shape)  # (224, 224, 3)
print("   NDVI Image shape:", ndvi_image.shape)  # (224, 224)
print("\n2. Batch Processing (typical training):")
print("   Batch of AG images: [batch_size, 224, 224, 3]")
print("   Batch of NDVI images: [batch_size, 224, 224, 1]")
print("\n3. Temporal Sequence (full growing season):")
print("   AG sequence: [num_images, 224, 224, 3]  # ~26 images per season")
print("   NDVI sequence: [num_images, 224, 224]")
print("\n4. Multi-modal combination:")
print("   Combined input: Concat(AG features, NDVI features)")
print("   -> Model extracts features from both modalities")
print()
print("="*70)
print("WHAT THE MODEL SEES:")
print("="*70)
print("""
The model will receive:
  1. Visual patterns in AG images:
     - Crop field boundaries and rows
     - Vegetation color (green intensity)
     - Soil types and moisture
     - Field management patterns
     
  2. Vegetation health from NDVI:
     - High values (0.7-1.0) = Healthy crops
     - Medium values (0.3-0.7) = Developing crops
     - Low values (-1-0.3) = Soil, drought, or harvest
     
  3. Temporal changes:
     - Planting phase: Low vegetation
     - Growth phase: Increasing NDVI
     - Maturation: Peak NDVI
     - Harvest: Sudden drop in NDVI
""")
print("="*70)

# Print summary info
print("\nSummary saved to visualization file!")
print(f"Check: cropnet_satellite_images_structure.png")
