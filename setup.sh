#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment..."
"$PYTHON" -m venv venv

echo "==> Installing dependencies..."
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -e .

echo "==> Creating import directory..."
mkdir -p import

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo ""
    echo "NOTE: .env created from .env.example — fill in your API keys before running."
else
    echo "==> .env already exists, skipping."
fi

echo ""
echo "Done. Run with:"
echo "  cd $(pwd) && venv/bin/python -m mixlab --genre drum_and_bass"
