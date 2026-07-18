#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Cow Anomaly Detection — Full Pipeline"
echo "============================================"
echo "Project : $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Create virtualenv if not present
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verify GPU
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('GPU: Apple MPS')
else:
    print('WARNING: No GPU detected, running on CPU')
"

# Run the full pipeline (forward all CLI args: --from-step, --force, --output-dir)
echo ""
echo "Starting pipeline..."
python3 -m scripts.run_full_pipeline "$@"

echo ""
echo "Pipeline finished."
