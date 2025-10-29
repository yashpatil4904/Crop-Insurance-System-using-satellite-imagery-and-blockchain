# GPU Setup Instructions

## Current Status
- **GPU Detected**: NVIDIA GeForce RTX 4050 Laptop GPU ✅
- **CUDA Available**: Version 12.7 ✅
- **PyTorch CUDA**: Not installed yet ❌

## Installation Steps

### Option 1: Manual Installation (Recommended)

1. **Close all Python programs** (including this one)

2. **Open a new terminal/command prompt**

3. **Run these commands one by one:**
   ```bash
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Wait for installation** (will download ~2.4 GB)

5. **Verify installation:**
   ```bash
   python check_gpu.py
   ```

### Option 2: Using the Install Script

1. **Close all Python programs**
2. **Double-click `install_cuda_pytorch.bat`**
3. **Wait for completion**
4. **Verify:** `python check_gpu.py`

## After Installation

Once CUDA PyTorch is installed, you can train with GPU:

```bash
python train_cropnet.py --data_dir ./cropnet_dataset --batch_size 64 --num_epochs 5 --lr 0.0005 --weather_processor lstm --save_dir ./cropnet_models
```

**Expected speedup**: 10-50x faster than CPU training!

## Troubleshooting

If you see "file locked" errors:
- Close ALL Python windows and IDEs
- Restart command prompt
- Try installation again

If GPU still not detected:
- Restart computer
- Verify nvidia-smi works
- Reinstall CUDA drivers if needed

