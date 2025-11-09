#!/bin/bash

# Multi-Topic Text Generator - Setup Script
# This script sets up both backend and frontend

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Multi-Topic Text Generator - Mixture-of-Experts Setup      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

print_info "Python version: $(python3 --version)"
print_info "Node.js version: $(node --version)"
echo ""

# Backend Setup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setting up Backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cp .env.example .env
else
    print_warning ".env file already exists, skipping..."
fi

print_info "Backend setup complete! ✓"
echo ""

cd ..

# Frontend Setup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setting up Frontend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd frontend

# Install Node dependencies
print_info "Installing Node.js dependencies (this may take a few minutes)..."
npm install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cp .env.example .env
else
    print_warning ".env file already exists, skipping..."
fi

print_info "Frontend setup complete! ✓"
echo ""

cd ..

# Final message
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup Complete! 🎉"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_info "To start the application:"
echo ""
echo "  1. Start Backend (Terminal 1):"
echo "     cd backend"
echo "     source venv/bin/activate"
echo "     python main.py"
echo ""
echo "  2. Start Frontend (Terminal 2):"
echo "     cd frontend"
echo "     npm run dev"
echo ""
echo "  3. Open your browser:"
echo "     Frontend: http://localhost:3000"
echo "     API Docs: http://localhost:8000/docs"
echo ""
print_info "Or use Docker Compose:"
echo "     docker-compose up"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
