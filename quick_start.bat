@echo off
REM Quick Start Training Script for Windows
REM Run this to get training results immediately!

echo.
echo ================================================================
echo                 POTHOLE DETECTION QUICK START
echo ================================================================
echo.

cd /d "%~dp0"

echo Step 1: Installing dependencies...
pip install torch torchvision opencv-python numpy matplotlib tqdm scikit-learn albumentations pyyaml requests

echo.
echo Step 2: Creating synthetic dataset (1000 samples)...
python scripts/prepare_datasets.py --synthetic --synthetic-samples 1000

echo.
echo Step 3: Quick training (10 epochs, mini model)...
python src/train.py --model mini --task classification --epochs 10 --batch-size 16 --quick-mode

echo.
echo Step 4: Testing the trained model...
python src/inference.py --model models/best_model.pth --test-mode

echo.
echo ================================================================
echo                        TRAINING COMPLETE!
echo ================================================================
echo Check results in:
echo   - logs/ for training plots and logs
echo   - models/ for saved models
echo.
echo Next steps:
echo   1. Download real pothole datasets for better accuracy
echo   2. Set up your Raspberry Pi hardware
echo   3. Collect local road images for training
echo.
pause