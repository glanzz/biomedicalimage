#!/bin/bash

# Setup script for Phase 1: Environment Setup
# Verifies installation and creates initial configuration

echo "=================================================="
echo "Biomedical Image Captioning - Environment Setup"
echo "Team 18: Shwetanshu & Bhargav"
echo "=================================================="

# Check Python version
echo -e "\n[1/6] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
echo -e "\n[2/6] Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "✗ Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# Install requirements
echo -e "\n[5/6] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "✓ Core dependencies installed"

    # Install additional packages
    pip install git+https://github.com/salaniz/pycocoevalcap.git -q
    echo "✓ pycocoevalcap installed"

    # Download NLTK data
    python -c "import nltk; nltk.download('punkt', quiet=True)"
    echo "✓ NLTK data downloaded"
else
    echo "✗ requirements.txt not found"
    exit 1
fi

# Verify installation
echo -e "\n[6/6] Verifying installation..."
python << 'EOF'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {display_name}: NOT INSTALLED")
        return False

print("\nCore Libraries:")
check_import('torch', 'PyTorch')
check_import('transformers', 'Transformers')
check_import('datasets', 'Datasets')
check_import('peft', 'PEFT')

print("\nData Processing:")
check_import('dask', 'Dask')
check_import('pandas', 'Pandas')
check_import('numpy', 'NumPy')

print("\nEvaluation:")
check_import('nltk', 'NLTK')
check_import('rouge_score', 'ROUGE')

print("\nUtilities:")
check_import('yaml', 'PyYAML')
check_import('matplotlib', 'Matplotlib')

print("\nGPU Support:")
import torch
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  (No GPU detected - training will be slower)")
EOF

# Verify directory structure
echo -e "\nDirectory Structure:"
for dir in data src scripts configs outputs slurm_jobs; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ - MISSING"
    fi
done

# Create necessary subdirectories
echo -e "\nCreating output directories..."
mkdir -p outputs/{checkpoints,logs,metrics,plots}
mkdir -p outputs/logs/tensorboard
mkdir -p slurm_jobs/logs
echo "✓ Output directories created"

# Summary
echo -e "\n=================================================="
echo "Environment Setup Complete!"
echo "=================================================="
echo ""
echo "Next Steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start Phase 2: python scripts/download_data.sh"
echo "  3. View implementation plan: IMPLEMENTATION_PLAN.md"
echo ""
echo "For cluster usage:"
echo "  - Edit configs/cluster_config.yaml with your username"
echo "  - Submit jobs: sbatch slurm_jobs/scripts/train_*.sh"
echo ""
