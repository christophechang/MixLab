#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-$(which python3.13 || which python3.12 || echo python3)}

echo "==> Creating virtual environment (.venv)..."
"$PYTHON" -m venv .venv

echo "==> Installing dependencies (editable + dev extras)..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]"

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
echo "  ./mixlab --genre house"
