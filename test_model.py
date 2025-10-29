#!/usr/bin/env python3
"""
Test/Inference script for trained CropNet model
Loads a trained model and makes predictions on new data
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from cropnet_model import CropNetModel
from cropnet_dataset import CropNetDataset
import matplotlib.pyplot as plt


def load_model(model_path, device, weather_processor_type='lstm'):
    """Load a trained model"""
    print(f"Loading model from {model_path}...")
    
    # Initialize model
    model = CropNetModel(weather_processor_type=weather_processor_type)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    if 'val_loss' in checkpoint:
        print(f"Model metrics from training:")
        print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Validation R²: {checkpoint.get('val_r2', 'N/A'):.4f}")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    
    return model, checkpoint


def test_on_dataset(model, test_loader, dataset, device, save_dir):
    """Test model on a dataset and save results"""
    print("\nEvaluating model on test set...")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            ag_image = batch['ag_image'].to(device)
            ndvi_image = batch['ndvi_image'].to(device)
            daily_weather = batch['daily_weather'].to(device)
            monthly_weather = batch['monthly_weather'].to(device)
            fips_idx = batch['fips_idx'].to(device)
            crop_idx = batch['crop_idx'].to(device)
            year_idx = batch['year_idx'].to(device)
            growth_stage = batch['growth_stage'].to(device)
            yield_target = batch['yield'].to(device)
            
            # Forward pass
            outputs = model(
                ag_image, ndvi_image, daily_weather, monthly_weather,
                fips_idx, crop_idx, year_idx, growth_stage
            )
            
            # Collect predictions
            all_preds.extend(outputs.squeeze().detach().cpu().numpy())
            all_targets.extend(yield_target.cpu().numpy())
            all_sample_ids.extend(batch['sample_id'].cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)
    
    # Calculate R²
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nTest Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    # Inverse transform if normalized
    if dataset.normalize_weather:
        all_preds_original = dataset.yield_scaler.inverse_transform(
            all_preds.reshape(-1, 1)
        ).flatten()
        all_targets_original = dataset.yield_scaler.inverse_transform(
            all_targets.reshape(-1, 1)
        ).flatten()
    else:
        all_preds_original = all_preds
        all_targets_original = all_targets
    
    # Save predictions
    save_dir = Path(save_dir)
    results = {
        'metrics': {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        },
        'predictions': [
            {
                'sample_id': int(sid),
                'true_yield': float(true),
                'pred_yield': float(pred)
            }
            for sid, true, pred in zip(all_sample_ids, all_targets_original, all_preds_original)
        ]
    }
    
    with open(save_dir / 'test_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPredictions saved to {save_dir / 'test_predictions.json'}")
    
    # Create visualization
    create_prediction_plot(all_targets_original, all_preds_original, r2, save_dir)
    
    return r2, rmse, mae


def create_prediction_plot(true_values, pred_values, r2, save_dir):
    """Create scatter plot of predictions vs. true values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(true_values, pred_values, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('True Yield')
    axes[0].set_ylabel('Predicted Yield')
    axes[0].set_title(f'Predictions vs. True Values (R² = {r2:.4f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residual plot
    residuals = true_values - pred_values
    axes[1].scatter(pred_values, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Yield')
    axes[1].set_ylabel('Residuals (True - Predicted)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'test_predictions_plot.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_dir / 'test_predictions_plot.png'}")
    plt.close()


def predict_single_sample(model, dataset, sample_idx, device):
    """Make prediction on a single sample"""
    sample = dataset[sample_idx]
    
    # Prepare inputs (add batch dimension)
    ag_image = sample['ag_image'].unsqueeze(0).to(device)
    ndvi_image = sample['ndvi_image'].unsqueeze(0).to(device)
    daily_weather = sample['daily_weather'].unsqueeze(0).to(device)
    monthly_weather = sample['monthly_weather'].unsqueeze(0).to(device)
    fips_idx = sample['fips_idx'].unsqueeze(0).to(device)
    crop_idx = sample['crop_idx'].unsqueeze(0).to(device)
    year_idx = sample['year_idx'].unsqueeze(0).to(device)
    growth_stage = sample['growth_stage'].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(
            ag_image, ndvi_image, daily_weather, monthly_weather,
            fips_idx, crop_idx, year_idx, growth_stage
        )
    
    # Inverse transform if normalized
    if dataset.normalize_weather:
        pred_value = dataset.yield_scaler.inverse_transform(
            prediction.cpu().numpy().reshape(-1, 1)
        )[0, 0]
        true_value = dataset.yield_scaler.inverse_transform(
            sample['yield'].numpy().reshape(-1, 1)
        )[0, 0]
    else:
        pred_value = prediction.item()
        true_value = sample['yield'].item()
    
    return pred_value, true_value


def main():
    parser = argparse.ArgumentParser(description='Test CropNet model')
    parser.add_argument('--model_path', type=str, default='./cropnet_models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='./cropnet_dataset',
                       help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='./cropnet_models',
                       help='Directory to save test results')
    parser.add_argument('--weather_processor', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Weather processor type (must match training)')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='Test on a single sample (optional)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.model_path, device, args.weather_processor)
    
    # Load dataset
    metadata_path = Path(args.data_dir) / 'dataset_metadata.json'
    dataset = CropNetDataset(metadata_path, normalize_weather=True)
    
    # Get test loader
    _, _, test_loader = dataset.get_loaders(
        batch_size=32,
        num_workers=4
    )
    
    # Test on single sample if specified
    if args.sample_idx is not None:
        print(f"\nTesting on sample {args.sample_idx}...")
        pred, true = predict_single_sample(model, dataset, args.sample_idx, device)
        print(f"  True Yield: {true:.2f}")
        print(f"  Predicted Yield: {pred:.2f}")
        print(f"  Error: {abs(pred - true):.2f}")
        print(f"  Error %: {abs(pred - true) / true * 100:.2f}%")
        return
    
    # Test on entire test set
    r2, rmse, mae = test_on_dataset(model, test_loader, dataset, device, args.save_dir)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE!")
    print(f"{'='*60}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == '__main__':
    main()

