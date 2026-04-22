"""
Retrain models with improved configuration for better accuracy.
"""

import subprocess
import sys

print("🚀 Retraining Car Object Detection Models with Improvements\n")
print("=" * 60)
print("\n✨ Improvements Applied:\n")

improvements = [
    "📈 Increased epochs: 15 → 50 (more training)",
    "🔧 Fine-tuning ResNet50: Unfreezing last 50 layers",
    "📊 Better dense architecture: (256, 128) → (512, 256, 128)",
    "🎯 Optimizer-specific learning rates + gradient clipping",
    "⏹️  Early stopping: Prevents overfitting",
    "📉 Learning rate scheduling: Reduces LR on plateau",
    "💾 Model checkpointing: Saves best validation model",
    "🛡️  Dropout regularization: 30% dropout in dense layers",
    "📌 L2 regularization: 1e-4 coefficient",
    "💪 Better SGD: Added Nesterov momentum (0.95)",
    "📐 Better localization objective: Huber + IoU-aware loss",
]

for improvement in improvements:
    print(f"  {improvement}")

print("\n" + "=" * 60)
print("\n⏳ Starting training... This may take 30-60 minutes.\n")

# Run main.py with improved parameters
cmd = [
    sys.executable,
    "main.py",
    "--epochs", "50",
    # --trainable-base is now the default
]

result = subprocess.run(cmd, cwd=".", capture_output=False)
sys.exit(result.returncode)
