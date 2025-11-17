#!/bin/bash

# Multi-Topic Text Generator - Backend Startup Script

echo "🚀 Starting Multi-Topic Text Generator Backend..."

# Check for virtual environment in parent directory (root) or current directory
if [ -d "../venv" ]; then
    VENV_PATH="../venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
else
    echo "❌ Virtual environment not found!"
    echo "📦 Creating virtual environment in parent directory..."
    cd ..
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv venv
    else
        echo "⚠️  Python 3.11 not found. Using system python3..."
        python3 -m venv venv
    fi
    VENV_PATH="venv"
    cd backend
fi

# Activate virtual environment
echo "🔌 Activating virtual environment from $VENV_PATH..."
source "$VENV_PATH/bin/activate"

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Using Python $PYTHON_VERSION"

# Check if Python 3.14 (which causes segfaults)
if [[ "$PYTHON_VERSION" == 3.14* ]]; then
    echo "⚠️  WARNING: Python 3.14 detected! This may cause segmentation faults."
    echo "   Please use Python 3.11 or 3.12 instead."
    echo "   Run: python3.11 -m venv ../venv"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists, if not copy from example
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from .env.example..."
    cp .env.example .env
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Starting FastAPI server..."
echo "📡 API will be available at: http://localhost:8000"
echo "📚 API docs will be available at: http://localhost:8000/docs"
echo ""

# Start the server
python main.py
