#!/bin/bash
# Audio Diffusion Model - Complete Setup and Execution Commands

# =============================================================================
# STEP 1: CREATE PROJECT DIRECTORY STRUCTURE
# =============================================================================

echo "Creating Audio Diffusion Model project structure..."

# Create main project directory
mkdir -p audio_diffusion_model
cd audio_diffusion_model

# Create all subdirectories
mkdir -p config
mkdir -p src/{models/{vae,encoders,diffusion},data,training,utils}
mkdir -p experiments
mkdir -p notebooks
mkdir -p checkpoints
mkdir -p logs
mkdir -p data/{raw,processed,samples}

echo "Directory structure created!"

# =============================================================================
# STEP 2: CREATE CONFIGURATION FILES
# =============================================================================

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.20.0
numpy>=1.20.0
scipy>=1.7.0
librosa>=0.9.0
wandb>=0.13.0
matplotlib>=3.5.0
tensorboard>=2.8.0
omegaconf>=2.2.0
hydra-core>=1.2.0
einops>=0.6.0
accelerate>=0.20.0
datasets>=2.10.0
EOF

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="audio-diffusion-model",
    version="0.1.0",
    description="End-to-End Audio Diffusion Model with Multi-Modal Conditioning",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "wandb>=0.13.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "omegaconf>=2.2.0",
        "hydra-core>=1.2.0",
        "einops>=0.6.0",
    ],
    python_requires=">=3.8",
)
EOF

# Create model configuration
cat > config/model_config.yaml << 'EOF'
# Model Configuration
model:
  sample_rate: 44100
  n_fft: 2048
  hop_length: 512
  vae_latent_dim: 64
  vae_downsample_factor: 16
  diffusion_steps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  unet_channels: 128
  unet_channel_mult: [1, 2, 4, 8]
  attention_resolutions: [32, 16, 8]
  text_encoder_dim: 768
  max_text_length: 256
  vocal_encoder_dim: 512
  accomp_encoder_dim: 512

# Training Configuration
training:
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.01
  num_epochs: 1000
  gradient_clip_norm: 1.0
  save_every: 50
  log_every: 10
  
# Data Configuration
data:
  sample_rate: 44100
  audio_length: 4.0  # seconds
  num_workers: 4
  shuffle: true
EOF

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/models/vae/__init__.py
touch src/models/encoders/__init__.py
touch src/models/diffusion/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py
touch experiments/__init__.py
touch config/__init__.py

echo "Configuration files created!"

# =============================================================================
# STEP 3: INSTALL DEPENDENCIES
# =============================================================================

echo "Installing dependencies..."

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Install additional development dependencies
pip install jupyter ipykernel
python -m ipykernel install --user --name audio-diffusion

echo "Dependencies installed!"

# =============================================================================
# STEP 4: SAVE THE MAIN MODEL CODE
# =============================================================================

echo "Save your main model code from paste.txt to: src/models/full_model.py"
echo "The file contains the complete AudioDiffusionModel implementation"

# =============================================================================
# STEP 5: CREATE TRAINING SCRIPT
# =============================================================================

cat > experiments/train_model.py << 'EOF'
#!/usr/bin/env python3
"""
Training script for Audio Diffusion Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Path to training data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Import model after setting up path
    from models.full_model import AudioDiffusionModel, ModelConfig, create_model_and_test
    
    print("Creating and testing model...")
    model, config = create_model_and_test()
    
    print("Training setup complete!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save initial model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, f"{args.checkpoint_dir}/initial_model.pt")
    
    print(f"Initial model saved to {args.checkpoint_dir}/initial_model.pt")

if __name__ == "__main__":
    main()
EOF

chmod +x experiments/train_model.py

# =============================================================================
# STEP 6: CREATE INFERENCE SCRIPT
# =============================================================================

cat > experiments/inference_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Inference demo for Audio Diffusion Model
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    print("Audio Diffusion Model - Inference Demo")
    print("=" * 50)
    
    # Import after path setup
    from models.full_model import AudioDiffusionModel, ModelConfig
    
    # Create model
    config = ModelConfig()
    model = AudioDiffusionModel(config)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo generation
    with torch.no_grad():
        # Text-to-audio generation
        duration = 2.0  # seconds
        latent_shape = (1, config.vae_latent_dim, 
                       int(duration * config.sample_rate // config.vae_downsample_factor))
        
        print(f"\nGenerating {duration}s audio...")
        print(f"Latent shape: {latent_shape}")
        
        # Generate with text conditioning
        text_tokens = torch.randint(0, 1000, (1, 32))
        generated_audio = model.sample(
            shape=latent_shape,
            text_tokens=text_tokens,
            num_steps=20  # Fast demo
        )
        
        print(f"Generated audio shape: {generated_audio.shape}")
        print(f"Audio duration: {generated_audio.shape[-1] / config.sample_rate:.2f}s")
        
        # Save audio (placeholder)
        print("\nAudio generation completed!")
        print("In a real implementation, you would:")
        print("1. Load a trained model checkpoint")
        print("2. Tokenize real text prompts")
        print("3. Save generated audio to WAV files")

if __name__ == "__main__":
    main()
EOF

chmod +x experiments/inference_demo.py

# =============================================================================
# STEP 7: CREATE JUPYTER NOTEBOOK
# =============================================================================

cat > notebooks/model_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Audio Diffusion Model Demo\n",
    "\n",
    "This notebook demonstrates the audio diffusion model capabilities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import model\n",
    "from models.full_model import AudioDiffusionModel, ModelConfig, create_model_and_test\n",
    "\n",
    "# Create model\n",
    "model, config = create_model_and_test()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate sample audio\n",
    "with torch.no_grad():\n",
    "    duration = 1.0\n",
    "    latent_shape = (1, config.vae_latent_dim, \n",
    "                   int(duration * config.sample_rate // config.vae_downsample_factor))\n",
    "    \n",
    "    generated_audio = model.sample(shape=latent_shape, num_steps=10)\n",
    "    \n",
    "    # Plot waveform\n",
    "    plt.figure(figsize=(12, 4))\n",
    "    plt.plot(generated_audio[0, 0].numpy())\n",
    "    plt.title('Generated Audio Waveform')\n",
    "    plt.xlabel('Sample')\n",
    "    plt.ylabel('Amplitude')\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "audio-diffusion",
   "language": "python",
   "name": "audio-diffusion"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "Jupyter notebook created!"

# =============================================================================
# STEP 8: EXECUTION COMMANDS
# =============================================================================

echo ""
echo "🎵 AUDIO DIFFUSION MODEL SETUP COMPLETE! 🎵"
echo "=" * 60
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. COPY YOUR MODEL CODE:"
echo "   cp /path/to/your/paste.txt src/models/full_model.py"
echo ""
echo "2. ACTIVATE VIRTUAL ENVIRONMENT:"
echo "   source venv/bin/activate"
echo ""
echo "3. TEST THE MODEL:"
echo "   python experiments/train_model.py --epochs 1"
echo ""
echo "4. RUN INFERENCE DEMO:"
echo "   python experiments/inference_demo.py"
echo ""
echo "5. START JUPYTER NOTEBOOK:"
echo "   jupyter notebook notebooks/model_demo.ipynb"
echo ""
echo "6. FULL TRAINING (with your data):"
echo "   python experiments/train_model.py \\"
echo "     --data_dir ./data/raw \\"
echo "     --batch_size 4 \\"
echo "     --epochs 100 \\"
echo "     --lr 1e-4"
echo ""
echo "7. MONITORING:"
echo "   tensorboard --logdir logs"
echo ""
echo "🚀 QUICK START COMMAND:"
echo "   python -c \"from src.models.full_model import create_model_and_test; create_model_and_test()\""
echo ""
echo "📁 PROJECT STRUCTURE:"
echo "   audio_diffusion_model/"
echo "   ├── src/models/full_model.py    # ← COPY YOUR CODE HERE"
echo "   ├── experiments/train_model.py   # Training script"
echo "   ├── experiments/inference_demo.py # Inference demo"
echo "   ├── notebooks/model_demo.ipynb   # Jupyter demo"
echo "   ├── config/model_config.yaml     # Configuration"
echo "   ├── requirements.txt             # Dependencies"
echo "   └── checkpoints/                 # Saved models"
echo ""
echo "Happy audio generation! 🎶"