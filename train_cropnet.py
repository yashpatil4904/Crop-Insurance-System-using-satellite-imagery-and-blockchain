#!/usr/bin/env python3
"""
Training script for CropNet model
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from cropnet_dataset import CropNetDataset
from cropnet_model import CropNetModel


def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, num_epochs=50, early_stopping_patience=10,
               save_dir='./models', validate_every_batch=True):
    """
    Train the CropNet model
    
    Args:
        model: CropNetModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        save_dir: Directory to save models
        validate_every_batch: Whether to validate after each batch
        
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectory for epoch models
    epoch_models_dir = save_dir / 'epoch_models'
    epoch_models_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize variables
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'batch_val_loss': []  # Track validation loss after each batch
    }
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
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
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                ag_image, ndvi_image, daily_weather, monthly_weather,
                fips_idx, crop_idx, year_idx, growth_stage
            )
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), yield_target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * ag_image.size(0)
            train_preds.extend(outputs.squeeze().detach().cpu().numpy())
            train_targets.extend(yield_target.cpu().numpy())
            
            # Validate after each batch if requested
            if validate_every_batch:
                batch_val_loss = validate_batch(model, val_loader, criterion, device)
                history['batch_val_loss'].append(batch_val_loss)
                
                # Print batch statistics
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {batch_val_loss:.4f}")
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Full validation phase
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion, device)
        val_r2 = r2_score(val_targets, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_r2': train_r2,
            'val_r2': val_r2,
        }, epoch_models_dir / f'model_epoch_{epoch+1}.pth')
        print(f"Saved model for epoch {epoch+1}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, save_dir / 'best_model.pth')
            print(f"Saved best model with val_loss: {val_loss:.4f}, val_r2: {val_r2:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final training history with batch validation
    with open(save_dir / 'detailed_training_history.json', 'w') as f:
        json.dump(history, f)
    
    return model, history


def validate_batch(model, val_loader, criterion, device):
    """Quick validation on a single batch"""
    model.eval()
    val_loss = 0.0
    
    # Get a single batch from validation loader
    val_iter = iter(val_loader)
    batch = next(val_iter)
    
    with torch.no_grad():
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
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), yield_target)
        val_loss = loss.item()
    
    return val_loss


def validate_model(model, val_loader, criterion, device):
    """Full validation on the entire validation set"""
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
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
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), yield_target)
            
            # Track statistics
            val_loss += loss.item() * ag_image.size(0)
            val_preds.extend(outputs.squeeze().detach().cpu().numpy())
            val_targets.extend(yield_target.cpu().numpy())
    
    # Calculate average loss
    val_loss = val_loss / len(val_loader.dataset)
    
    return val_loss, val_preds, val_targets
    
    # This is a duplicate of the final model save - removing
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def r2_score(y_true, y_pred):
    """Calculate R² score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Avoid division by zero
    if ss_tot == 0:
        return 0
    
    return 1 - (ss_res / ss_tot)


def plot_training_history(history, save_dir):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot R²
    ax2.plot(history['train_r2'], label='Train')
    ax2.plot(history['val_r2'], label='Validation')
    ax2.set_title('R² Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png')
    plt.close()


def evaluate_model(model, test_loader, criterion, device, dataset, save_dir='./models'):
    """
    Evaluate the model on the test set
    
    Args:
        model: Trained CropNetModel
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        dataset: CropNetDataset instance (for inverse transform)
        save_dir: Directory to save results (can be str or Path)
        
    Returns:
        Test loss and R² score
    """
    # Convert save_dir to Path if it's a string
    save_dir = Path(save_dir)
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_targets = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
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
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), yield_target)
            
            # Track statistics
            test_loss += loss.item() * ag_image.size(0)
            test_preds.extend(outputs.squeeze().detach().cpu().numpy())
            test_targets.extend(yield_target.cpu().numpy())
            sample_ids.extend(batch['sample_id'].cpu().numpy())
    
    # Calculate statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"Test Loss: {test_loss:.4f}, R²: {test_r2:.4f}")
    
    # Inverse transform predictions and targets if normalized
    if dataset.normalize_weather:
        test_preds = dataset.yield_scaler.inverse_transform(
            np.array(test_preds).reshape(-1, 1)
        ).flatten()
        test_targets = dataset.yield_scaler.inverse_transform(
            np.array(test_targets).reshape(-1, 1)
        ).flatten()
    
    # Save predictions (convert numpy types to native Python types)
    results = {
        'sample_id': [int(sid) for sid in sample_ids],
        'true_yield': [float(ty) for ty in test_targets],
        'pred_yield': [float(py) for py in test_preds],
    }
    
    with open(save_dir / 'test_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot predictions vs. targets
    plt.figure(figsize=(10, 8))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], 
             [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('True Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Test Set: True vs. Predicted Yield (R² = {test_r2:.4f})')
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / 'test_predictions.png')
    plt.close()
    
    return test_loss, test_r2


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CropNet model')
    parser.add_argument('--data_dir', type=str, default='./cropnet_dataset',
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--weather_processor', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Weather processor type')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataset
    metadata_path = Path(args.data_dir) / 'dataset_metadata.json'
    dataset = CropNetDataset(metadata_path, normalize_weather=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = CropNetModel(weather_processor_type=args.weather_processor)
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        # Update device to use first GPU
        device = torch.device('cuda:0')
        model = model.to(device)
    
    # Print model summary
    print(f"\nCropNet Model Summary:")
    print(f"  Weather Processor: {args.weather_processor}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Create loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    # Note: verbose parameter removed as it's not supported in this PyTorch version
    
    # Train model
    print(f"\nStarting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=args.num_epochs, early_stopping_patience=args.patience,
        save_dir=args.save_dir, validate_every_batch=False  # Disabled for speed
    )
    
    # Evaluate model
    print(f"\nEvaluating model on test set...")
    test_loss, test_r2 = evaluate_model(
        model, test_loader, criterion, device, dataset, save_dir=args.save_dir
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation R²: {max(history['val_r2']):.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")


if __name__ == '__main__':
    main()
