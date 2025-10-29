"""
Simple CropNet Record Demo
Shows the structure of a CropNet record without HRRR (which requires GRIB libraries)
"""

import sys
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime

# Install basic packages
def pip_install(packages):
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

print("Installing packages...")
pip_install(["cropnet", "matplotlib", "pillow", "numpy", "requests"])

# Import
try:
    from cropnet.data_downloader import DataDownloader
    from cropnet.data_retriever import DataRetriever
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configuration
FIPS = ["10003"]  # New Castle County, Delaware
YEAR = "2022"
TARGET_DIR = "./data"
os.makedirs(TARGET_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"CropNet Record Demo")
print(f"County FIPS: {FIPS[0]} (New Castle County, DE)")
print(f"Year: {YEAR}")
print(f"{'='*60}\n")

# Step 1: Download data
print("1. Downloading CropNet data...")
print("   (This may take a few minutes)")

downloader = DataDownloader(target_dir=TARGET_DIR)

try:
    # USDA yield data
    print("   - Downloading USDA Soybean yield data...")
    downloader.download_USDA("Soybean", fips_codes=FIPS, years=[YEAR])
    
    # Sentinel-2 imagery
    print("   - Downloading Sentinel-2 AG images...")
    downloader.download_Sentinel2(fips_codes=FIPS, years=[YEAR], image_type="AG")
    
    print("   - Downloading Sentinel-2 NDVI images...")
    downloader.download_Sentinel2(fips_codes=FIPS, years=[YEAR], image_type="NDVI")
    
    print("[OK] Download complete!")
except Exception as e:
    print(f"   [WARN] Download issue: {e}")
    print("   -> Will try to retrieve existing data...")

# Step 2: Retrieve and build record
print("\n2. Building unified record...")

retriever = DataRetriever(base_dir=TARGET_DIR)

# Initialize record structure
record = {
    "metadata": {
        "fips_code": FIPS[0],
        "county_name": "New Castle County, DE",
        "year": YEAR,
        "crop_type": "Soybean"
    },
    "usda_yield": None,
    "sentinel2_images": {
        "ag_path": None,
        "ndvi_path": None,
        "ag_shape": None,
        "ndvi_shape": None
    },
    "hrrr_weather": {
        "note": "HRRR requires GRIB libraries (eccodes) - skipped for Windows compatibility",
        "alternative": "Weather data would contain: temperature, humidity, precipitation, etc."
    }
}

# Retrieve USDA data
try:
    usda = retriever.retrieve_USDA(crop_type="Soybean", fips_codes=FIPS, years=[YEAR])
    if usda and len(usda) > 0:
        usda_values = list(usda.values())
        if isinstance(usda_values[0], list) and len(usda_values[0]) > 0:
            record["usda_yield"] = usda_values[0][0].get("yield", "N/A")
            print(f"   - USDA Yield: {record['usda_yield']}")
except Exception as e:
    print(f"   [WARN] USDA retrieval: {e}")

# Retrieve Sentinel-2 almost everything
try:
    s2_ag = retriever.retrieve_Sentinel2(fips_codes=FIPS, years=[YEAR], image_type="AG")
    if s2_ag and len(s2_ag) > 0:
        ag_data = list(s2_ag.values())[0]
        if isinstance(ag_data, list) and len(ag_data) > 0:
            record["sentinel2_images"]["ag_path"] = str(ag_data[0].get("path", "N/A"))
            if os.path.exists(record["sentinel2_images"]["ag_path"]):
                img = Image.open(record["sentinel2_images"]["ag_path"])
                record["sentinel2_images"]["ag_shape"] = list(img.size)
            print(f"   - AG Image: {record['sentinel2_images']['ag_path']}")
except Exception as e:
    print(f"   [WARN] AG retrieval: {e}")

try:
    s2_ndvi = retriever.retrieve_Sentinel2(fips_codes=FIPS, years=[YEAR], image_type="NDVI")
    if s2_ndvi and len(s2_ndvi) > 0:
        ndvi_data = list(s2_ndvi.values())[0]
        if isinstance(ndvi_data, list) and len(ndvi_data) > 0:
            record["sentinel2_images"]["ndvi_path"] = str(ndvi_data[0].get("path", "N/A"))
            if os.path.exists(record["sentinel2_images"]["ndvi_path"]):
                img = Image.open(record["sentinel2_images"]["ndvi_path"])
                record["sentinel2_images"]["ndvi_shape"] = list(img.size)
            print(f"   - NDVI Image: {record['sentinel2_images']['ndvi_path']}")
except Exception as e:
    print(f"   [WARN] NDVI retrieval: {e}")

# Display the record
print("\n3. CropNet Unified Record Structure:")
print("="*60)
print(json.dumps(record, indent=2, default=str))
print("="*60)

# Step 3: Visualize images
print("\n4. Loading and displaying Sentinel-2 images...")

ag_path = record["sentinel2_images"]["ag_path"]
ndvi_path = record["sentinel2_images"]["ndvi_path"]

if ag_path and ag_path != "N/A" and os.path.exists(ag_path):
    try:
        ag_img = Image.open(ag_path)
        print(f"   [OK] AG image loaded: {ag_img.size}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # AG image
        axes[0].imshow(ag_img)
        axes[0].set_title(f"Sentinel-2 Agriculture Imagery\nFIPS: {FIPS[0]}, Year: {YEAR}", fontsize=11)
        axes[0].axis("off")
        
        # NDVI image
        if ndvi_path and ndvi_path != "N/A" and os.path.exists(ndvi_path):
            ndvi_img = Image.open(ndvi_path)
            axes[1].imshow(ndvi_img, cmap="viridis")
            axes[1].set_title("Sentinel-2 NDVI", fontsize=11)
            print(f"   [OK] NDVI image loaded: {ndvi_img.size}")
        else:
            axes[1].text(0.5, 0.5, "NDVI\nNot Available", ha='center', va='center', fontsize=14)
            axes[1].set_title("Sentinel-2 NDVI", fontsize=11)
        
        axes[1].axis("off")
        plt.suptitle("CropNet Dataset: Sentinel-2 Satellite Imagery", fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_file = "cropnet_record_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   [OK] Visualization saved to '{output_file}'")
        plt.show()
        
    except Exception as e:
        print(f"   [WARN] Image visualization failed: {e}")
else:
    print("   [WARN] No AG image available for visualization")

# Summary
print("\n5. Summary:")
print(f"   County: {record['metadata']['county_name']}")
print(f"   FIPS: {record['metadata']['fips_code']}")
print(f"   Year: {record['metadata']['year']}")
print(f"   Yield: {record['usda_yield']}")
print(f"   AG Image: {'Available' if record['sentinel2_images']['ag_path'] and record['sentinel2_images']['ag_path'] != 'N/A' else 'Not available'}")
print(f"   NDVI Image: {'Available' if record['sentinel2_images']['ndvi_path'] and record['sentinel2_images']['ndvi_path'] != 'N/A' else 'Not available'}")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
