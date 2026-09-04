#!/bin/bash
#SBATCH --job-name=GPUAllatom
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --account=phy230061p

# -----------------------------
# Load Anaconda
# -----------------------------
module purge

# Load personal Miniconda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd "$SLURM_SUBMIT_DIR"

ENV_NAME="mace_clean"
ENV_PATH="$HOME/miniconda3/envs/$ENV_NAME"

echo "=== Conda Diagnostics ==="
echo "HOME=$HOME"
echo "ENV_PATH=$ENV_PATH"

echo "Conda:"
which conda
conda --version

conda info
conda config --show envs_dirs

# -----------------------------
# Create environment if missing
# -----------------------------
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda environment: $ENV_NAME"

    conda create -y -p "$ENV_PATH" python=3.10 pip || exit 1

    conda install -y -p "$ENV_PATH" -c conda-forge \
        libstdcxx-ng \
        libgcc-ng || exit 1

    conda run -p "$ENV_PATH" python -m pip install --upgrade pip || exit 1

    # Install CUDA-enabled PyTorch
    conda run -p "$ENV_PATH" python -m pip install \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 || exit 1

    # Install remaining packages
    conda run -p "$ENV_PATH" python -m pip install \
        mace-torch \
        ase \
        numpy \
        scipy \
        pandas \
        matplotlib || exit 1
else
    echo "Using existing environment:"
    echo "$ENV_PATH"
fi

# -----------------------------
# Activate environment
# -----------------------------
conda activate "$ENV_PATH" || exit 1

# -----------------------------
# Runtime fixes
# -----------------------------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -----------------------------
# Diagnostics
# -----------------------------
echo "Python:"
which python
python --version

echo "Packages:"
python -c "import torch, scipy, mace; print('torch', torch.__version__); print('scipy', scipy.__version__); print('mace', mace.__version__)"

echo "=== NODE INFO ==="
hostname

echo "=== GPU INFO ==="
nvidia-smi || echo "NO GPU FOUND"

echo "=== PYTORCH CUDA CHECK ==="
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

python -c "import torch; print('CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# -----------------------------
# Run your program
# -----------------------------
python MACEB2.py
