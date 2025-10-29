# CropNet Project - Main Files

## Core Essential Files (Must Keep)

### 1. Model Architecture
- **`cropnet_model.py`** - Defines the CropNet model architecture
  - AG CNN branch (ResNet18 for RGB images)
  - NDVI CNN branch (ResNet18 for vegetation index)
  - Weather LSTM/Transformer processor
  - Feature fusion and regression layers

### 2. Dataset Handling
- **`cropnet_dataset.py`** - PyTorch Dataset class for loading data
  - Loads AG and NDVI images
  - Processes weather data and metadata
  - Creates train/val/test splits
  - Handles data normalization

### 3. Training Script
- **`train_cropnet.py`** - Main training script
  - Trains the model
  - Validates performance
  - Saves checkpoints
  - Tracks training history

### 4. Testing/Inference Script
- **`test_model.py`** - Model evaluation and prediction script
  - Loads trained models
  - Evaluates on test set
  - Makes predictions on new data
  - Generates performance metrics

### 5. Dataset Generation
- **`create_cropnet_dataset.py`** - Creates synthetic CropNet dataset
  - Generates 2,500 samples
  - Creates AG and NDVI images
  - Generates weather data
  - Saves dataset structure

## Supporting Files

### Documentation
- **`README.md`** - Project overview and usage instructions
- **`SETUP_INSTRUCTIONS.md`** - GPU setup guide
- **`.gitignore`** - Git ignore rules

### Visualization (Optional)
- **`architecture_diagram.py`** - Generates model architecture diagram

## Data Files

### Dataset
- **`cropnet_dataset/`** - Complete dataset directory
  - `dataset_metadata.json` - All sample records with weather/yield data
  - `dataset_summary.json` - Statistics by crop type
  - `images/` - AG and NDVI image files (.npy format)

### Example Data
- **`cropnet_record_example.json`** - Example record structure

## Generated Files (After Training)

### Models
- **`cropnet_models/`** - Trained models and results
  - `best_model.pth` - Best performing model
  - `epoch_models/` - Models saved after each epoch
  - `test_predictions.json` - Test set predictions
  - `test_predictions_plot.png` - Prediction visualization
  - `training_history.json` - Training metrics

## Minimal File Set

If you want the absolute minimum to run the project:

1. **`cropnet_model.py`** - Model definition
2. **`cropnet_dataset.py`** - Data loading
3. **`train_cropnet.py`** - Training
4. **`test_model.py`** - Testing
5. **`create_cropnet_dataset.py`** - Dataset creation
6. **`cropnet_dataset/`** - Dataset directory

All other files are optional or generated.

