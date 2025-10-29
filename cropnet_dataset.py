#!/usr/bin/env python3
"""
CropNet Dataset: PyTorch Dataset for loading CropNet data
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd


class CropNetDataset(Dataset):
    """
    PyTorch Dataset for CropNet data
    
    Loads AG images, NDVI images, weather data, and yield values
    for crop yield prediction
    """
    def __init__(self, metadata_path, transform=None, normalize_weather=True, 
                 fips_mapping=None, crop_mapping=None):
        """
        Initialize the dataset
        
        Args:
            metadata_path: Path to dataset_metadata.json
            transform: Optional transforms to apply to images
            normalize_weather: Whether to normalize weather data
            fips_mapping: Dict mapping FIPS codes to indices
            crop_mapping: Dict mapping crop types to indices
        """
        self.metadata_path = Path(metadata_path)
        self.transform = transform
        self.normalize_weather = normalize_weather
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples")
        
        # Create mappings if not provided
        if fips_mapping is None:
            self.fips_mapping = self._create_fips_mapping()
        else:
            self.fips_mapping = fips_mapping
            
        if crop_mapping is None:
            self.crop_mapping = self._create_crop_mapping()
        else:
            self.crop_mapping = crop_mapping
            
        # Create year mapping (assuming years 2017-2022)
        self.year_mapping = {year: idx for idx, year in enumerate(range(2017, 2023))}
        
        # Initialize weather scalers
        if normalize_weather:
            self._fit_weather_scalers()
    
    def _create_fips_mapping(self):
        """Create mapping from FIPS codes to indices"""
        unique_fips = sorted(set(sample['fips'] for sample in self.data))
        return {fips: idx for idx, fips in enumerate(unique_fips)}
    
    def _create_crop_mapping(self):
        """Create mapping from crop types to indices"""
        unique_crops = sorted(set(sample['crop_type'] for sample in self.data))
        return {crop: idx for idx, crop in enumerate(unique_crops)}
    
    def _fit_weather_scalers(self):
        """Fit scalers for weather data normalization"""
        # Extract all daily weather data
        daily_data = []
        for sample in self.data:
            daily_data.extend(sample['hrrr_daily'])
        daily_array = np.array(daily_data)
        
        # Create and fit daily scaler
        self.daily_scaler = StandardScaler()
        self.daily_scaler.fit(daily_array)
        
        # Extract and process monthly weather data
        monthly_data = []
        for sample in self.data:
            # Extract numerical values from monthly data
            monthly_values = []
            for month in sample['hrrr_monthly']:
                monthly_values.extend([
                    month['avg_temperature'],
                    month['total_precipitation'],
                    month['growing_degree_days']
                ])
            monthly_data.append(monthly_values)
        monthly_array = np.array(monthly_data)
        
        # Create and fit monthly scaler
        self.monthly_scaler = StandardScaler()
        self.monthly_scaler.fit(monthly_array)
        
        # Fit yield scaler
        yields = np.array([sample['usda_yield'] for sample in self.data]).reshape(-1, 1)
        self.yield_scaler = StandardScaler()
        self.yield_scaler.fit(yields)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.data[idx]
        
        # Load images
        ag_image = np.load(sample['ag_image_path'])
        ndvi_image = np.load(sample['ndvi_image_path'])
        
        # Process AG image (RGB)
        # Convert from [H, W, C] to [C, H, W] for PyTorch
        ag_image = np.transpose(ag_image, (2, 0, 1)).astype(np.float32)
        
        # Process NDVI image (single channel)
        # Add channel dimension for PyTorch [H, W] -> [1, H, W]
        ndvi_image = ndvi_image.astype(np.float32)[np.newaxis, :, :]
        
        # Apply transforms if any
        if self.transform:
            ag_image = self.transform(ag_image)
            ndvi_image = self.transform(ndvi_image)
        
        # Process daily weather data
        daily_weather = np.array(sample['hrrr_daily'], dtype=np.float32)
        if self.normalize_weather:
            daily_weather = self.daily_scaler.transform(daily_weather).astype(np.float32)
        
        # Process monthly weather data
        monthly_values = []
        for month in sample['hrrr_monthly']:
            monthly_values.extend([
                month['avg_temperature'],
                month['total_precipitation'],
                month['growing_degree_days']
            ])
        monthly_weather = np.array(monthly_values, dtype=np.float32)
        if self.normalize_weather:
            monthly_weather = self.monthly_scaler.transform(monthly_weather.reshape(1, -1))[0].astype(np.float32)
        
        # Get metadata
        fips_idx = self.fips_mapping[sample['fips']]
        crop_idx = self.crop_mapping[sample['crop_type']]
        year_idx = self.year_mapping[sample['year']]
        growth_stage = sample['growth_stage']
        
        # Get yield (target)
        yield_value = sample['usda_yield']
        if self.normalize_weather:
            yield_value = self.yield_scaler.transform([[yield_value]])[0][0].astype(np.float32)
        else:
            yield_value = np.float32(yield_value)
        
        # Convert to tensors
        ag_image_tensor = torch.from_numpy(ag_image)
        ndvi_image_tensor = torch.from_numpy(ndvi_image)
        daily_weather_tensor = torch.from_numpy(daily_weather)
        monthly_weather_tensor = torch.from_numpy(monthly_weather)
        fips_idx_tensor = torch.tensor(fips_idx, dtype=torch.long)
        crop_idx_tensor = torch.tensor(crop_idx, dtype=torch.long)
        year_idx_tensor = torch.tensor(year_idx, dtype=torch.long)
        growth_stage_tensor = torch.tensor(growth_stage, dtype=torch.float32)
        yield_tensor = torch.tensor(yield_value, dtype=torch.float32)
        
        return {
            'ag_image': ag_image_tensor,
            'ndvi_image': ndvi_image_tensor,
            'daily_weather': daily_weather_tensor,
            'monthly_weather': monthly_weather_tensor,
            'fips_idx': fips_idx_tensor,
            'crop_idx': crop_idx_tensor,
            'year_idx': year_idx_tensor,
            'growth_stage': growth_stage_tensor,
            'yield': yield_tensor,
            'sample_id': sample['sample_id']
        }
    
    def get_loaders(self, batch_size=32, val_split=0.15, test_split=0.15, 
                   random_state=42, num_workers=4):
        """
        Create train/val/test data loaders
        
        Args:
            batch_size: Batch size for dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            random_state: Random seed for splitting
            num_workers: Number of workers for dataloaders
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # Create indices for train/val/test split
        indices = np.arange(len(self))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        test_size = int(test_split * len(self))
        val_size = int(val_split * len(self))
        train_size = len(self) - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Create subset samplers
        from torch.utils.data import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            self, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"Created data loaders:")
        print(f"  Train: {train_size} samples ({len(train_loader)} batches)")
        print(f"  Validation: {val_size} samples ({len(val_loader)} batches)")
        print(f"  Test: {test_size} samples ({len(test_loader)} batches)")
        
        return train_loader, val_loader, test_loader

