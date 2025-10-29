#!/usr/bin/env python3
"""
Display one complete record and show how it's used for model training
"""

import json
import numpy as np
from pathlib import Path

print("="*70)
print("COMPLETE CROPNET RECORD")
print("="*70)

# Load the first record
with open('cropnet_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

sample = data[0]

# Display the complete record structure
print("\n1. COMPLETE RECORD STRUCTURE:")
print("-"*70)
print(json.dumps(sample, indent=2)[:3000])  # First part
print("\n... (truncated for display)")

# Load the actual images
print("\n\n2. IMAGE DATA:")
print("-"*70)
ag_image = np.load(sample['ag_image_path'])
ndvi_image = np.load(sample['ndvi_image_path'])

print(f"AG Image:")
print(f"  Shape: {ag_image.shape}")
print(f"  Dtype: {ag_image.dtype}")
print(f"  Range: [{np.min(ag_image):.3f}, {np.max(ag_image):.3f}]")
print(f"  Mean: {np.mean(ag_image):.3f}")

print(f"\nNDVI Image:")
print(f"  Shape: {ndvi_image.shape}")
print(f"  Dtype: {ndvi_image.dtype}")
print(f"  Range: [{np.min(ndvi_image):.3f}, {np.max(ndvi_image):.3f}]")
print(f"  Mean: {np.mean(ndvi_image):.3f}")

# Weather data
print("\n\n3. WEATHER DATA (HRRR):")
print("-"*70)
print(f"Daily weather shape: {len(sample['hrrr_daily'])} days x {len(sample['hrrr_daily'][0])} features")
print(f"Total daily values: {len(sample['hrrr_daily']) * len(sample['hrrr_daily'][0])}")
print(f"\nFirst 3 days of weather data:")

for i in range(min(3, len(sample['hrrr_daily']))):
    day = sample['hrrr_daily'][i]
    print(f"\nDay {i+1}:")
    print(f"  Temp Max: {day[0]:.2f}F, Min: {day[1]:.2f}F")
    print(f"  Precipitation: {day[2]:.3f} in")
    print(f"  Humidity: {day[3]:.2f}%")
    print(f"  Wind: {day[4]:.2f} mph")
    print(f"  Solar: {day[5]:.2f} W/m2")
    print(f"  Cloud Cover: {day[6]:.2f}%")
    print(f"  Soil Moisture: {day[7]:.3f}")
    print(f"  Soil Temp: {day[8]:.2f}F")

print(f"\nMonthly aggregates: {len(sample['hrrr_monthly'])} months")
if sample['hrrr_monthly']:
    print("\nFirst month:")
    m = sample['hrrr_monthly'][0]
    print(f"  Month: {m['month']}")
    print(f"  Avg Temp: {m['avg_temperature']:.2f}F")
    print(f"  Total Precip: {m['total_precipitation']:.2f} in")
    print(f"  Growing Degree Days: {m['growing_degree_days']:.2f}")

# Yield
print("\n\n4. TARGET VARIABLE (YIELD):")
print("-"*70)
print(f"USDA Yield: {sample['usda_yield']}")

print("\n\n5. HOW THIS RECORD IS USED FOR MODEL TRAINING:")
print("-"*70)
print("""
INPUT TO MODEL:
--------------
X (Image Features):
  - AG Image: Shape (224, 224, 3) -> Flatten to (150528,) or use CNN
  - NDVI Image: Shape (224, 224) -> Flatten to (50176,) or use CNN
  
Y_s (Short-term Weather Features):
  - Daily HRRR: Shape (183, 9) -> Flatten to (1647,) or use LSTM/RNN
  
Y_l (Long-term Weather Features):
  - Monthly HRRR: Shape (6, 3) -> Flatten to (18,)
  
METADATA (Optional Features):
  - FIPS code: Categorical
  - Year: Numerical
  - Crop Type: Categorical (one-hot encoded)
  - Growth Stage: Numerical (0-1)

TARGET (What Model Predicts):
--------------
Z (Yield):
  - USDA Yield: Real number (bushels/acre or lbs/acre)

MODEL ARCHITECTURE EXAMPLE:
--------------
1. CNN branches for images (AG + NDVI)
2. LSTM/RNN for daily weather sequence
3. Dense layer for monthly weather
4. Concatenate all features
5. Final Dense layer -> yield prediction

LOSS FUNCTION:
--------------
Mean Squared Error (MSE) or Mean Absolute Error (MAE)

TRAINING EXAMPLE:
--------------
(X, Y_s, Y_l) -> Model -> Predicted_Yield
                   |
                Compare with Ground Truth
                   |
                Compute Loss
                   |
                Backpropagate
""")

print("\n\n6. TRANSFORMED INPUT FOR MODEL:")
print("-"*70)

# Show how to convert to model input
ag_input = ag_image.reshape(-1)  # Flatten
ndvi_input = ndvi_image.reshape(-1)  # Flatten
weather_daily = np.array(sample['hrrr_daily'])
weather_monthly = sample['hrrr_monthly'][0]  # First month example

print("Flat AG Image:", ag_input.shape, "values")
print("Flat NDVI Image:", ndvi_input.shape, "values")
print("Daily Weather Array:", weather_daily.shape, "= (days, features)")
print("Monthly Weather Dict:", list(weather_monthly.keys()))

print("\n" + "="*70)
print("END OF RECORD")
print("="*70)

