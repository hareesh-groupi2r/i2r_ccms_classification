#!/bin/bash

# CCMS Backend Run Script
# Simple script to start the Flask development server

set -e

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "   Creating from template..."
    cp .env.example .env
    echo "   Please configure your .env file with API keys before running again"
    exit 1
fi

echo "🚀 Starting CCMS Backend Service..."
echo "   Server will be available at: http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""

# Activate virtual environment and run
source .venv/bin/activate
cd api
python app.py