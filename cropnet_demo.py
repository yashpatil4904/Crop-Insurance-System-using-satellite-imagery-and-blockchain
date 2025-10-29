#!/usr/bin/env python3
"""
CropNet Demo: Build one unified record and show structure
This script downloads minimal CropNet data and assembles a single record.
"""

import sys
import subprocess
import os
import json
from pathlib import Path

def pip_install(packages):
    """Install packages with fallback"""
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"[OK] Installed {pkg}")
        except subprocess.CalledProcessError:
            print(f"[FAIL] Failed to install {pkg}")

# Install required packages (skip ecmwflibs for Windows compatibility)
print("Installing packages...")
pip_install(["cropnet", "matplotlib", "pillow", "numpy"])

# Import after installation
try:
    from cropnet.data_downloader import DataDownloader
    from cropnet.data_retriever import DataRetriever
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Configuration
FIPS = ["10003"]  # New Castle County, Delaware
YEARS = ["2022"]
TARGET_DIR = "./data"
os.makedirs(TARGET_DIR, exist_ok=True)

print(f"\n=== CropNet Demo: FIPS {FIPS[0]}, Year {YEARS[0]} ===")

# Download data
print("\n1. Downloading data...")
downloader = DataDownloader(target_dir=TARGET_DIR)

try:
    # USDA yield data
    print("  - Downloading USDA Soybean data...")
    downloader.download_USDA("Soybean", fips_codes=FIPS, years=YEARS)
    
    # Sentinel-2 imagery
    print("  - Downloading Sentinel-2 AG images...")
    downloader.download_Sentinel2(fips_codes=FIPS, years=YEARS, image_type="AG")
    
    print("  - Downloading Sentinel-2 NDVI images...")
    downloader.download_Sentinel2(fips_codes=FIPS, years=YEARS, image_type="NDVI")
    
    # HRRR weather data (this might fail on Windows without ecmwflibs)
    print("  - Downloading HRRR weather data...")
    try:
        downloader.download_HRRR(fips_codes=FIPS, years=YEARS)
        print("  [OK] HRRR download successful")
    except Exception as e:
        print(f"  [WARN] HRRR download failed (expected on Windows): {e}")
        print("  -> Continuing without HRRR data...")
    
    print("[OK] Download phase complete")
    
except Exception as e:
    print(f"[FAIL] Download failed: {e}")
    print("-> Creating mock data structure instead...")

# Retrieve and assemble record
print("\n2. Assembling unified record...")
retriever = DataRetriever(base_dir=TARGET_DIR)

record = {
    "fips": FIPS[0],
    "year": YEARS[0],
    "crop_type": "Soybean",
    "usda_yield": None,
    "hrrr_features": None,
    "sentinel2_ag_path": None,
    "sentinel2_ndvi_path": None,
    "record_structure": {
        "description": "Unified CropNet record combining USDA yield, HRRR weather, and Sentinel-2 imagery",
        "fields": {
            "fips": "County FIPS code",
            "year": "Data year",
            "crop_type": "Crop type (Soybean, Corn, Cotton, Winter Wheat)",
            "usda_yield": "Ground truth yield from USDA (bushels/acre)",
            "hrrr_features": "Daily weather features [temp, humidity, precipitation, etc.]",
            "sentinel2_ag_path": "Path to Agriculture RGB image (224x224)",
            "sentinel2_ndvi_path": "Path to NDVI vegetation index image (224x224)"
        }
    }
}

# Try to populate with real data
try:
    # USDA data
    usda = retriever.retrieve_USDA(crop_type="Soybean", fips_codes=FIPS, years=YEARS)
    if usda and len(usda) > 0:
        usda_data = list(usda.values())[0]
        if isinstance(usda_data, list) and len(usda_data) > 0:
            record["usda_yield"] = usda_data[0].get("yield", "N/A")
    
    # Sentinel-2 data
    s2_ag = retriever.retrieve_Sentinel2(fips_codes=FIPS, years=YEARS, image_type="AG")
    s2_ndvi = retriever.retrieve_Sentinel2(fips_codes=FIPS, years=YEARS, image_type="NDVI")
    
    if s2_ag and len(s2_ag) > 0:
        ag_data = list(s2_ag.values())[0]
        if isinstance(ag_data, list) and len(ag_data) > 0:
            record["sentinel2_ag_path"] = str(ag_data[0].get("path", "N/A"))
    
    if s2_ndvi and len(s2_ndvi) > 0:
        ndvi_data = list(s2_ndvi.values())[0]
        if isinstance(ndvi_data, list) and len(ndvi_data) > 0:
            record["sentinel2_ndvi_path"] = str(ndvi_data[0].get("path", "N/A"))
    
    # HRRR data (if available)
    try:
        hrrr = retriever.retrieve_HRRR(fips_codes=FIPS, years=YEARS)
        if hrrr and len(hrrr) > 0:
            hrrr_data = list(hrrr.values())[0]
            if isinstance(hrrr_data, dict) and "daily" in hrrr_data:
                record["hrrr_features"] = f"Daily weather array with {len(hrrr_data['daily'])} days"
    except:
        record["hrrr_features"] = "Not available (Windows compatibility issue)"

except Exception as e:
    print(f"[WARN] Data retrieval error: {e}")
    print("-> Using mock structure...")

# Display the record
print("\n3. Unified CropNet Record:")
print("=" * 50)
print(json.dumps(record, indent=2))

# Try to visualize images if available
print("\n4. Image Visualization:")
ag_path = record["sentinel2_ag_path"]
ndvi_path = record["sentinel2_ndvi_path"]

if ag_path and ag_path != "N/A" and os.path.exists(ag_path):
    try:
        ag_img = Image.open(ag_path)
        print(f"[OK] AG Image loaded: {ag_img.size}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(ag_img)
        axes[0].set_title(f"Sentinel-2 AG\nFIPS: {FIPS[0]}, Year: {YEARS[0]}")
        axes[0].axis("off")
        
        if ndvi_path and ndvi_path != "N/A" and os.path.exists(ndvi_path):
            ndvi_img = Image.open(ndvi_path)
            axes[1].imshow(ndvi_img, cmap="viridis")
            axes[1].set_title("Sentinel-2 NDVI")
            print(f"[OK] NDVI Image loaded: {ndvi_img.size}")
        else:
            axes[1].text(0.5, 0.5, "NDVI\nNot Available", ha='center', va='center')
            axes[1].set_title("Sentinel-2 NDVI")
        
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("cropnet_images.png", dpi=150, bbox_inches='tight')
        print("[OK] Images saved to 'cropnet_images.png'")
        plt.show()
        
    except Exception as e:
        print(f"[WARN] Image visualization failed: {e}")
else:
    print("[WARN] No AG image available for visualization")

print("\n5. Summary:")
print(f"   • County: {FIPS[0]} (New Castle County, DE)")
print(f"   • Year: {YEARS[0]}")
print(f"   • Crop: {record['crop_type']}")
print(f"   • Yield: {record['usda_yield']}")
print(f"   • Weather: {record['hrrr_features']}")
print(f"   • AG Image: {'[OK]' if record['sentinel2_ag_path'] and record['sentinel2_ag_path'] != 'N/A' else '[MISSING]'}")
print(f"   • NDVI Image: {'[OK]' if record['sentinel2_ndvi_path'] and record['sentinel2_ndvi_path'] != 'N/A' else '[MISSING]'}")

print("\n=== Demo Complete ===")
