#!/usr/bin/env python3
"""
CropNet Model: Two-branch CNN architecture for crop yield prediction
- AG CNN branch
- NDVI CNN branch
- Weather MLP with temporal processing (LSTM/Transformer)
- Feature fusion and yield regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class AgCnnBranch(nn.Module):
    """CNN branch for processing AG (RGB) images"""
    def __init__(self, pretrained=True):
        super(AgCnnBranch, self).__init__()
        # Use ResNet18 as backbone, pretrained on ImageNet
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Output feature dimension is 512
        self.feature_dim = 512
        
    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        x = self.features(x)
        # Flatten: [batch_size, 512, 1, 1] -> [batch_size, 512]
        x = torch.flatten(x, 1)
        return x


class NdviCnnBranch(nn.Module):
    """CNN branch for processing NDVI images"""
    def __init__(self, pretrained=False):
        super(NdviCnnBranch, self).__init__()
        # Start with a ResNet but modify first layer for single-channel input
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # Replace first conv layer to accept single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If using pretrained weights, copy weights for first channel
        if pretrained:
            # Average RGB channels to create single channel weights
            weight = resnet.conv1.weight.data
            self.conv1.weight.data = weight.mean(dim=1, keepdim=True)
        
        # Use the rest of ResNet except first conv and final FC
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Output feature dimension is 512
        self.feature_dim = 512
        
    def forward(self, x):
        # x shape: [batch_size, 1, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        # Flatten: [batch_size, 512, 1, 1] -> [batch_size, 512]
        x = torch.flatten(x, 1)
        return x


class WeatherLstmProcessor(nn.Module):
    """LSTM-based processor for weather time series data"""
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, dropout=0.2):
        super(WeatherLstmProcessor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Output feature dimension is 2*hidden_dim due to bidirectional
        self.feature_dim = hidden_dim * 2
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        output, (hidden, _) = self.lstm(x)
        
        # Concatenate the final forward and backward hidden states
        # hidden shape: [num_layers*2, batch_size, hidden_dim]
        # Get last layer's hidden states
        hidden_forward = hidden[-2, :, :]  # Forward direction of last layer
        hidden_backward = hidden[-1, :, :]  # Backward direction of last layer
        
        # Concatenate: [batch_size, hidden_dim*2]
        x = torch.cat((hidden_forward, hidden_backward), dim=1)
        return x


class WeatherTransformerProcessor(nn.Module):
    """Transformer-based processor for weather time series data"""
    def __init__(self, input_dim=9, d_model=128, nhead=8, num_layers=2, dropout=0.2):
        super(WeatherTransformerProcessor, self).__init__()
        
        # Linear projection to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output feature dimension
        self.feature_dim = d_model
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        
        # Project input to d_model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Global pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MonthlyWeatherProcessor(nn.Module):
    """MLP for processing monthly weather aggregates"""
    def __init__(self, input_dim=18, hidden_dim=64):
        super(MonthlyWeatherProcessor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.feature_dim = hidden_dim
        
    def forward(self, x):
        # x shape: [batch_size, features]
        return self.mlp(x)


class MetadataProcessor(nn.Module):
    """Processor for metadata features (FIPS, year, crop type, growth stage)"""
    def __init__(self, num_fips=100, num_crops=4, num_years=6, embedding_dim=16, output_dim=64):
        super(MetadataProcessor, self).__init__()
        
        # Embeddings for categorical features
        self.fips_embedding = nn.Embedding(num_fips, embedding_dim)
        self.crop_embedding = nn.Embedding(num_crops, embedding_dim)
        self.year_embedding = nn.Embedding(num_years, embedding_dim)
        
        # MLP for combining embeddings and continuous features
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*3 + 1, output_dim),  # +1 for growth_stage
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
        self.feature_dim = output_dim
        
    def forward(self, fips_idx, crop_idx, year_idx, growth_stage):
        # Get embeddings
        fips_emb = self.fips_embedding(fips_idx)
        crop_emb = self.crop_embedding(crop_idx)
        year_emb = self.year_embedding(year_idx)
        
        # Concatenate embeddings with growth_stage
        growth_stage = growth_stage.unsqueeze(1)  # Add dimension
        x = torch.cat([fips_emb, crop_emb, year_emb, growth_stage], dim=1)
        
        # Process through MLP
        x = self.mlp(x)
        return x


class CropNetModel(nn.Module):
    """
    Complete CropNet model with:
    - AG CNN branch
    - NDVI CNN branch
    - Weather LSTM/Transformer
    - Feature fusion
    - Yield regression
    """
    def __init__(self, weather_processor_type='lstm'):
        super(CropNetModel, self).__init__()
        
        # Image processing branches
        self.ag_branch = AgCnnBranch(pretrained=True)
        self.ndvi_branch = NdviCnnBranch(pretrained=True)
        
        # Weather processing
        if weather_processor_type == 'lstm':
            self.weather_processor = WeatherLstmProcessor(input_dim=9, hidden_dim=128)
        else:  # transformer
            self.weather_processor = WeatherTransformerProcessor(input_dim=9, d_model=128)
            
        # Monthly weather processing
        self.monthly_processor = MonthlyWeatherProcessor(input_dim=18, hidden_dim=64)
        
        # Metadata processing
        self.metadata_processor = MetadataProcessor(output_dim=64)
        
        # Calculate total feature dimension
        total_feature_dim = (
            self.ag_branch.feature_dim +
            self.ndvi_branch.feature_dim +
            self.weather_processor.feature_dim +
            self.monthly_processor.feature_dim +
            self.metadata_processor.feature_dim
        )
        
        # Feature fusion and regression
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Final regression output
        )
        
    def forward(self, ag_img, ndvi_img, daily_weather, monthly_weather, 
                fips_idx, crop_idx, year_idx, growth_stage):
        """
        Forward pass through the complete model
        
        Args:
            ag_img: [batch_size, 3, 224, 224] - RGB satellite images
            ndvi_img: [batch_size, 1, 224, 224] - NDVI images
            daily_weather: [batch_size, seq_len, 9] - Daily weather features
            monthly_weather: [batch_size, 18] - Monthly weather features
            fips_idx: [batch_size] - FIPS code indices
            crop_idx: [batch_size] - Crop type indices
            year_idx: [batch_size] - Year indices
            growth_stage: [batch_size] - Growth stage values
            
        Returns:
            yield_pred: [batch_size, 1] - Predicted yield values
        """
        # Process each modality
        ag_features = self.ag_branch(ag_img)
        ndvi_features = self.ndvi_branch(ndvi_img)
        weather_features = self.weather_processor(daily_weather)
        monthly_features = self.monthly_processor(monthly_weather)
        metadata_features = self.metadata_processor(fips_idx, crop_idx, year_idx, growth_stage)
        
        # Concatenate all features
        combined_features = torch.cat([
            ag_features, ndvi_features, weather_features, 
            monthly_features, metadata_features
        ], dim=1)
        
        # Fusion and regression
        yield_pred = self.fusion(combined_features)
        
        return yield_pred

