#!/bin/bash
# =============================================================================
# setup.sh — Installation du projet sur OVHCloud AI Notebooks
# Usage : bash setup.sh
# =============================================================================

set -e

echo "========================================"
echo " Single-Source Fast Generation — Setup"
echo "========================================"

# --------------------------------------------------------------------------
# 0. Réinstaller PyTorch compatible avec le driver CUDA présent
# --------------------------------------------------------------------------
echo ""
echo "[0/5] Correction de PyTorch..."

# Récupère la version CUDA du driver (ex: "12.8")
CUDA_DRIVER=$(python -c "
import subprocess, re
try:
    out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL).decode()
    m = re.search(r'CUDA Version: (\S+)', out)
    print(m.group(1) if m else '0')
except:
    print('0')
")
echo "  Driver CUDA détecté : $CUDA_DRIVER"

# Désinstaller le PyTorch pré-installé (potentiellement incompatible)
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# torch 2.5.1+cu118 : disponible sur PyPI, fonctionne avec tout driver >= 12.x
# (cu118 = runtime CUDA 11.8 bundlé, le driver hôte n'a pas besoin d'être 11.8)
pip install --quiet \
    torch==2.6.0+cu118 \
    torchvision==0.21.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "  PyTorch 2.6.0+cu118 installé."

# --------------------------------------------------------------------------
# 1. Vérifier que CUDA est disponible
# --------------------------------------------------------------------------
echo ""
echo "[1/5] Vérification GPU..."
python -c "
import torch
print(f'  PyTorch         : {torch.__version__}')
print(f'  CUDA disponible : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ATTENTION : CUDA non détecté.')
    print('  → Lance : nvidia-smi  pour vérifier que le GPU est bien alloué.')
    exit(1)
"

# --------------------------------------------------------------------------
# 2. Initialiser le submodule SinFusion
# --------------------------------------------------------------------------
echo ""
echo "[2/5] Initialisation du submodule SinFusion..."
git submodule update --init --recursive
echo "  SinFusion disponible dans ./sinfusion/"

# --------------------------------------------------------------------------
# 3. Installer les dépendances Python
# --------------------------------------------------------------------------
echo ""
echo "[3/5] Installation des dépendances..."

# pytorch_lightning 1.5.10 a des métadonnées invalides pour pip>=24.1
# 1.9.5 = dernière version 1.x, même API (gpus=, etc.), compatible torch 2.x
pip install --quiet \
    einops==0.6.0 \
    "pytorch_lightning==1.9.5" \
    imageio==2.16.1 \
    scikit-image \
    lpips \
    tqdm \
    tensorboard

echo "  Dépendances installées."

# --------------------------------------------------------------------------
# 4. Vérifier que SinFusion s'importe correctement
# --------------------------------------------------------------------------
echo ""
echo "[4/5] Vérification SinFusion..."
cd sinfusion
python -c "
import sys
sys.path.insert(0, '.')
from models.nextnet import NextNet
from diffusion.diffusion import Diffusion
from diffusion.conditional_diffusion import ConditionalDiffusion
print('  NextNet              : OK')
print('  Diffusion            : OK')
print('  ConditionalDiffusion : OK')
"
cd ..

# --------------------------------------------------------------------------
# 5. Créer les dossiers de travail
# --------------------------------------------------------------------------
echo ""
echo "[5/5] Création des dossiers..."
mkdir -p data/images data/videos outputs checkpoints

echo ""
echo "========================================"
echo " Setup terminé avec succès !"
echo "========================================"
echo ""
echo " Lance l'entraînement SinFusion avec :"
echo "   cd sinfusion"
echo "   python main.py task=image image_name=balloons run_name=balloons_teacher"
echo ""
