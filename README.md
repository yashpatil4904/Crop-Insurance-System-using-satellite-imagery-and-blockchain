# CropNet: Crop Yield Prediction Model

A deep learning model for crop yield prediction using multi-modal data:
- Satellite imagery (AG RGB + NDVI)
- Weather time series
- Crop metadata

## Architecture

The model uses a two-branch architecture:
- **AG CNN Branch**: Processes RGB satellite images using ResNet18
- **NDVI CNN Branch**: Processes vegetation index images using modified ResNet18
- **Weather Processing**: LSTM or Transformer for temporal weather data
- **Feature Fusion**: Combines all modalities for yield prediction

![Architecture](architecture_diagram.png)

## Dataset

The dataset contains:
- 2,500 samples across 4 crop types (Corn, Soybean, Winter Wheat, Cotton)
- AG images: RGB satellite imagery (224×224×3)
- NDVI images: Vegetation index (224×224)
- Daily weather: 183 days × 9 features (temperature, precipitation, etc.)
- Monthly weather: 6 months of aggregated climate data
- Yield values: Ground truth for training

## Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib tqdm scikit-learn
```

### Training

```bash
python train_cropnet.py --data_dir ./cropnet_dataset --batch_size 32 --num_epochs 50
```

Options:
- `--weather_processor`: Choose between 'lstm' (default) or 'transformer'
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Maximum number of epochs (default: 50)
- `--lr`: Initial learning rate (default: 0.001)
- `--save_dir`: Directory to save models (default: ./models)

## Model Components

- `cropnet_model.py`: Model architecture definition
- `cropnet_dataset.py`: PyTorch dataset for loading data
- `train_cropnet.py`: Training and evaluation script

## Results

The model is evaluated using:
- Mean Squared Error (MSE) loss
- R² score

Example results:
- Training R²: 0.XX
- Validation R²: 0.XX
- Test R²: 0.XX

## Citation

```
@misc{cropnet2023,
  author = {Your Name},
  title = {CropNet: Crop Yield Prediction Using Multi-Modal Deep Learning},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/cropnet}}
}
```

