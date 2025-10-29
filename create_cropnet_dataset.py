"""
CropNet Dataset Generator
Creates a realistic dataset with 2500 samples covering all crop types and US counties
"""

import numpy as np
import json
import os
from pathlib import Path
from itertools import cycle

# Crop types in CropNet dataset
CROP_TYPES = ["Corn", "Soybean", "Winter Wheat", "Cotton"]

# Sample FIPS codes from different states (real US counties)
# These represent major agricultural regions
SAMPLE_FIPS = {
    "Corn": ["17097", "18089", "19013", "27003", "27037", "38017", "38071", "39049", "39103", "46077"],
    "Soybean": ["17097", "18089", "19013", "27003", "27037", "38017", "38071", "39049", "39103", "46077"],
    "Winter Wheat": ["20015", "20077", "20111", "40017", "40031", "48093", "48249", "53167", "56013"],
    "Cotton": ["01009", "01011", "05017", "22007", "35025", "48179", "48249", "48261"]
}

# Years in dataset
YEARS = list(range(2017, 2023))  # 2017-2022 (6 years)

print("="*70)
print("CROPNET DATASET GENERATOR")
print("="*70)
print("\nDataset Configuration:")
print(f"  - Total samples: 2500")
print(f"  - Crop types: {CROP_TYPES}")
print(f"  - Years: {YEARS[0]}-{YEARS[-1]}")
print(f"  - Image size: 224x224")
print("="*70)

# Configuration
N_SAMPLES = 2500
IMG_SIZE = 224
SAVE_DIR = Path("./cropnet_dataset")

# Clean up old dataset if exists
if SAVE_DIR.exists():
    import shutil
    shutil.rmtree(SAVE_DIR)
    print("[INFO] Removed old dataset")

SAVE_DIR.mkdir(exist_ok=True)

print("\nGenerating 2500 realistic CropNet samples...")
print("This may take a few minutes...\n")

# Initialize dataset structure
dataset = []

def generate_realistic_ag_image(crop_type, year, growth_stage):
    """
    Generate realistic AG (Agriculture RGB) image
    Simulates crop fields with proper patterns
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    
    # Base soil color (light brown)
    base_soil = np.array([0.6, 0.55, 0.5])  # RGB
    
    # Generate field patterns
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Create crop row patterns
            row_pattern = np.sin(i * 0.1) * 0.3 + 0.7
            
            # Growth stage affects color intensity
            if growth_stage < 0.3:  # Early planting
                vegetation = 0.3 + np.random.uniform(-0.1, 0.1)
            elif growth_stage < 0.7:  # Growing
                vegetation = 0.5 + np.random.uniform(-0.1, 0.1) * row_pattern
            else:  # Mature
                vegetation = 0.8 + np.random.uniform(-0.05, 0.1) * row_pattern
            
            # Apply crop-specific colors
            if crop_type == "Corn":
                # Corn is dark green when healthy
                img[i, j] = [
                    vegetation * 0.2,      # Red
                    vegetation * 0.7,      # Green
                    vegetation * 0.15      # Blue
                ]
            elif crop_type == "Soybean":
                # Soybeans are medium green
                img[i, j] = [
                    vegetation * 0.25,
                    vegetation * 0.65,
                    vegetation * 0.2
                ]
            elif crop_type == "Winter Wheat":
                # Wheat is lighter green/yellow
                img[i, j] = [
                    vegetation * 0.4,
                    vegetation * 0.6,
                    vegetation * 0.1
                ]
            elif crop_type == "Cotton":
                # Cotton is lighter green
                img[i, j] = [
                    vegetation * 0.35,
                    vegetation * 0.6,
                    vegetation * 0.25
                ]
    
    # Add field boundaries (dark lines)
    for i in range(0, IMG_SIZE, 40):
        img[i:i+2, :, :] = base_soil * 0.5
    for j in range(0, IMG_SIZE, 40):
        img[:, j:j+2, :] = base_soil * 0.5
    
    return np.clip(img, 0, 1)

def generate_realistic_ndvi_image(crop_type, year, growth_stage):
    """
    Generate realistic NDVI image
    Values range from -1 to 1 (typical NDVI range)
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    # Base NDVI based on growth stage
    base_ndvi = -0.2 + growth_stage * 1.2  # From bare soil to healthy vegetation
    
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Add field patterns
            row_pattern = np.sin(i * 0.1) * 0.3 + 1.0
            
            # NDVI varies with growth
            if growth_stage < 0.3:  # Early
                ndvi = 0.1 + np.random.uniform(-0.1, 0.1)
            elif growth_stage < 0.7:  # Growing
                ndvi = 0.4 + np.random.uniform(-0.15, 0.15) * row_pattern
            else:  # Mature
                ndvi = 0.75 + np.random.uniform(-0.1, 0.1) * row_pattern
            
            img[i, j] = ndvi
    
    # Add field boundaries (low NDVI)
    for i in range(0, IMG_SIZE, 40):
        img[i:i+2, :] = -0.1
    for j in range(0, IMG_SIZE, 40):
        img[:, j:j+2] = -0.1
    
    return np.clip(img, -1, 1)

def generate_hrrr_weather(year, days_in_season=183):
    """
    Generate realistic HRRR weather features
    9 weather parameters per day
    """
    # Daily features (9 parameters)
    daily_features = []
    
    # Typical growing season is April-September (~180 days)
    months = ['04', '05', '06', '07', '08', '09']
    
    for month_idx, month in enumerate(cycle(months[:len(months)])):
        if len(daily_features) >= days_in_season:
            break
        
        # Weather varies by month
        if month in ['04', '05']:  # Spring
            base_temp = 55 + np.random.uniform(-5, 10)
            base_precip = 0.1 + np.random.exponential(0.15)
        elif month in ['06', '07', '08']:  # Summer
            base_temp = 75 + np.random.uniform(-5, 10)
            base_precip = 0.15 + np.random.exponential(0.2)
        else:  # Fall
            base_temp = 60 + np.random.uniform(-5, 8)
            base_precip = 0.1 + np.random.exponential(0.15)
        
        # 9 weather parameters
        day_features = [
            base_temp,  # temperature_max (Fahrenheit)
            base_temp - 20 + np.random.uniform(0, 10),  # temperature_min
            base_precip,  # precipitation (inches)
            60 + np.random.uniform(-10, 20),  # humidity (%)
            7 + np.random.uniform(-3, 5),  # wind_speed (mph)
            400 + np.random.uniform(-100, 100),  # solar_radiation (W/m²)
            35 + np.random.uniform(-10, 30),  # cloud_cover (%)
            0.35 + np.random.uniform(-0.1, 0.15),  # soil_moisture
            base_temp - 25 + np.random.uniform(0, 10)  # soil_temperature
        ]
        
        daily_features.append(day_features)
    
    # Monthly aggregates
    monthly_features = []
    for month in months:
        monthly = {
            'month': month,
            'avg_temperature': 65 + np.random.uniform(-5, 10),
            'total_precipitation': 3.5 + np.random.uniform(-1, 2),
            'growing_degree_days': 200 + np.random.uniform(0, 100)
        }
        monthly_features.append(monthly)
    
    return {
        'daily': daily_features[:days_in_season],
        'monthly': monthly_features
    }

def get_yield_ground_truth(crop_type, weather_quality):
    """
    Estimate realistic yield based on crop type and weather
    Using typical US averages as baselines
    """
    # Baseline yields (bushels/acre for most, lbs/acre for cotton)
    baselines = {
        "Corn": 175.0,  # bushels/acre
        "Soybean": 48.0,  # bushels/acre
        "Winter Wheat": 50.0,  # bushels/acre
        "Cotton": 850.0  # lbs/acre
    }
    
    # Apply weather influence
    weather_factor = 0.8 + weather_quality * 0.4  # 0.8 to 1.2 range
    yield_value = baselines[crop_type] * weather_factor
    
    # Add some realistic variance
    yield_value += np.random.uniform(-5, 5) * baselines[crop_type] * 0.05
    
    return round(yield_value, 2)

# Main dataset generation loop
sample_idx = 0
samples_per_crop = N_SAMPLES // len(CROP_TYPES)

for crop_type in CROP_TYPES:
    print(f"\nGenerating {samples_per_crop} samples for {crop_type}...")
    
    for crop_sample in range(samples_per_crop):
        # Select random FIPS and year
        fips = np.random.choice(SAMPLE_FIPS[crop_type])
        year = np.random.choice(YEARS)
        
        # Random growth stage (0-1, simulating different times in growing season)
        growth_stage = np.random.uniform(0.3, 1.0)  # Assume decent crop conditions
        
        # Generate images
        ag_image = generate_realistic_ag_image(crop_type, year, growth_stage)
        ndvi_image = generate_realistic_ndvi_image(crop_type, year, growth_stage)
        
        # Generate weather
        weather_data = generate_hrrr_weather(year, days_in_season=183)
        
        # Calculate weather quality (average of normalized features)
        avg_daily = np.mean(weather_data['daily'], axis=0)
        weather_quality = np.mean([avg_daily[0] / 100,  # normalized temp
                                   avg_daily[1] / 100,
                                   1 - min(avg_daily[2], 1),  # less precip is better
                                   avg_daily[4] / 15,  # wind
                                   1 - min(avg_daily[6] / 100, 1)  # cloud cover
                                   ])
        
        # Get yield ground truth
        yield_value = get_yield_ground_truth(crop_type, weather_quality)
        
        # Create sample record
        sample = {
            'sample_id': int(sample_idx),
            'fips': fips,
            'state': fips[:2] if len(fips) >= 2 else "00",
            'county_code': fips[2:] if len(fips) >= 2 else fips,
            'year': int(year),
            'crop_type': crop_type,
            'growth_stage': round(growth_stage, 3),
            
            # Images (saved as paths, actual arrays saved separately)
            'ag_image_path': f"{SAVE_DIR}/images/ag_{crop_type}_{fips}_{year}_{sample_idx}.npy",
            'ndvi_image_path': f"{SAVE_DIR}/images/ndvi_{crop_type}_{fips}_{year}_{sample_idx}.npy",
            
            # Weather data (cast to pure Python types)
            'hrrr_daily': np.asarray(weather_data['daily'], dtype=float).tolist(),
            'hrrr_monthly': weather_data['monthly'],
            
            # Ground truth
            'usda_yield': yield_value,
            'weather_quality': round(weather_quality, 3)
        }
        
        # Save images
        image_dir = SAVE_DIR / "images"
        image_dir.mkdir(exist_ok=True)
        np.save(sample['ag_image_path'], ag_image)
        np.save(sample['ndvi_image_path'], ndvi_image)
        
        dataset.append(sample)
        sample_idx += 1
        
        if (sample_idx % 100) == 0:
            print(f"  Progress: {sample_idx}/{N_SAMPLES} samples generated...")

print(f"\n[OK] Generated {len(dataset)} samples!")

# Save metadata
metadata_path = SAVE_DIR / "dataset_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"[OK] Saved metadata to {metadata_path}")

# Create summary statistics
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)

stats = {}
for crop in CROP_TYPES:
    crop_samples = [s for s in dataset if s['crop_type'] == crop]
    stats[crop] = {
        'count': len(crop_samples),
        'avg_yield': float(np.mean([s['usda_yield'] for s in crop_samples])) if crop_samples else 0.0,
        'std_yield': float(np.std([s['usda_yield'] for s in crop_samples])) if crop_samples else 0.0
    }
    print(f"\n{crop}:")
    print(f"  Samples: {stats[crop]['count']}")
    print(f"  Avg Yield: {stats[crop]['avg_yield']:.2f}")
    print(f"  Std Yield: {stats[crop]['std_yield']:.2f}")

# Save summary
summary_path = SAVE_DIR / "dataset_summary.json"
with open(summary_path, 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*70)
print("[OK] DATASET GENERATION COMPLETE!")
print("="*70)
print(f"\nOutput directory: {SAVE_DIR}")
print(f"  - {len(dataset)} sample records")
print(f"  - {len(dataset) * 2} image files (AG + NDVI)")
print(f"  - Ready for PyTorch Dataset wrapper")
print("\nNext step: Create PyTorch Dataset class for model training")
