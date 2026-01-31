# 🚀 Quick Start Guide - Pneumo AI

Get up and running in 5 minutes!

## ⚡ One-Click Setup (Windows)

We've provided a helper script to automate installation and launch:

```powershell
./start_app.ps1
```

This will:
1. Install all dependencies
2. Launch the Web UI

---

## Manual Setup

## Prerequisites

- Python 3.9+
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA 11.8+

## Step 1: Installation (2 minutes)

```bash
# Navigate to project
cd Pneumo-AI

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Step 2: Download Dataset (10 minutes)

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle account)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract
unzip chest-xray-pneumonia.zip -d data/

# Prepare CSV files
python scripts/prepare_dataset.py --data-dir data/chest_xray --output-dir data
```

**Expected output:**
```
data/
├── images/          # 5,863 X-ray images
├── train.csv        # ~4,100 samples
├── val.csv          # ~880 samples
└── test.csv         # ~880 samples
```

## Step 3: Train Model (2-4 hours on GPU, 12+ hours on CPU)

```bash
# Start training
python train.py --config configs/config.yaml

# Monitor progress
# Training logs will show:
# - Epoch progress
# - Loss, accuracy, recall per epoch
# - Best model saved to checkpoints/
```

**Expected checkpoints:**
```
checkpoints/
└── pneumonia_detector_best_val_recall.pth
```

## Step 4: Evaluate Model (5 minutes)

```bash
# Run evaluation
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/pneumonia_detector_best_val_recall.pth \
  --output-dir evaluation_results

# View results
cat evaluation_results/metrics.json
```

**Expected metrics:**
- Recall: >0.90
- Precision: >0.80
- F1 Score: >0.85

## Step 5: Run Web UI (1 minute)

```bash
# Launch Streamlit app
streamlit run app/streamlit_app.py

# Open browser to http://localhost:8501
```

**What you'll see:**
- 🫁 Medical-themed interface
- 📤 Upload chest X-ray
- 🔬 Click "Analyze X-Ray"
- 📊 View prediction + heatmap

## Step 6: Test Inference (1 minute)

```bash
# Single image prediction
python predict.py \
  --image data/test/NORMAL2-IM-0001-0001.jpeg \
  --checkpoint checkpoints/pneumonia_detector_best_val_recall.pth \
  --save-heatmap \
  --output results/prediction.json

# View results
cat results/prediction.json
```

---

## 🐳 Docker Quick Start (Alternative)

If you prefer Docker:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  pneumo-ai

# Open http://localhost:8501
```

---

## 🎯 What's Next?

### For Research/Development
1. **Experiment**: Modify `configs/config.yaml` and retrain
2. **Ensemble**: Train EfficientNet-B4 and combine models
3. **Fine-tune**: Adjust thresholds for target recall

### For Production Deployment
1. **Validate**: Run clinical validation study
2. **Optimize**: Export to ONNX for faster inference
3. **Deploy**: Use Docker Compose for production
4. **Monitor**: Set up performance tracking

---

## 📚 Key Files to Know

- **`configs/config.yaml`** - All system parameters
- **`train.py`** - Training script
- **`predict.py`** - Inference script
- **`app/streamlit_app.py`** - Web UI
- **`README.md`** - Full documentation

---

## ⚠️ Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

### Slow Training
```bash
# Disable AMP if causing issues
training:
  use_amp: false
```

### Model Not Found
```bash
# Check checkpoint path
ls checkpoints/
# Update path in config.yaml or command line
```

---

## 🆘 Need Help?

- **Documentation**: See [README.md](README.md)
- **Examples**: Check [notebooks/demo.md](notebooks/demo.md)
- **Issues**: Open GitHub issue

---

**You're ready to detect pneumonia! 🫁**
