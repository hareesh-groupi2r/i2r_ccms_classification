#!/bin/bash

# CCMS Backend Setup Script
# Simple one-command setup for new developers

set -e  # Exit on any error

echo "🚀 Setting up CCMS Backend Python Service..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.12"

if [ "$python_version" != "$required_version" ]; then
    echo "⚠️  Warning: Python $required_version recommended, found $python_version"
    echo "   The setup will continue but you may encounter compatibility issues."
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   Virtual environment already exists, removing..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "✅ Virtual environment created at .venv/"

# Activate virtual environment and upgrade pip
echo "🔧 Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
.venv/bin/python -m pip install -r requirements.txt
