#!/bin/bash

# LAMBDA Framework Installation Script
# This script helps you set up the LAMBDA framework environment

set -e  # Exit on any error

echo "=================================="
echo "LAMBDA Framework Installation"
echo "=================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected successfully"
else
    echo "❌ Error: Python 3.7 or higher required, current version: $python_version"
    exit 1
fi

# Check Java version
echo "Checking Java version..."
if command -v java &> /dev/null; then
    java_version=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1,2)
    echo "✅ Java $java_version detected successfully"
else
    echo "❌ Warning: Java not detected, please install Java 8 or higher"
    echo "   Download link: https://adoptopenjdk.net/"
fi

# Check Maven
echo "Checking Maven..."
if command -v mvn &> /dev/null; then
    maven_version=$(mvn --version | head -n1 | cut -d' ' -f3)
    echo "✅ Maven $maven_version detected successfully"
else
    echo "❌ Warning: Maven not detected, please install Maven"
    echo "   Download link: https://maven.apache.org/download.cgi"
fi

# Create virtual environment (optional but recommended)
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "lambda_env" ]; then
    python3 -m venv lambda_env
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source lambda_env/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "Installing basic dependencies..."
    pip install javalang>=0.13.0 openai>=1.0.0 anthropic>=0.18.0 beautifulsoup4>=4.9.0 lxml>=4.6.0 requests>=2.25.0 numpy>=1.19.0 pandas>=1.3.0
    echo "✅ Basic dependencies installed successfully"
fi

# Install package in development mode
echo "Installing LAMBDA Framework..."
pip install -e .

# Create necessary directories
echo "Creating necessary directory structure..."
mkdir -p results/static_analysis
mkdir -p results/test_generation
mkdir -p results/bug_reports
mkdir -p logs
echo "✅ Directory structure created successfully"

# Set environment variables
echo "Setting environment variables..."
export JAVA_OPTS="-Xmx4g"
echo "export JAVA_OPTS=\"-Xmx4g\"" >> ~/.bashrc

echo ""
echo "=================================="
echo "Installation completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source lambda_env/bin/activate"
echo "2. Configure API keys (if using LLM features):"
echo "   export OPENAI_API_KEY='your-api-key'"
echo "   export ANTHROPIC_API_KEY='your-api-key'"
echo "3. Run example:"
echo "   python run.py /path/to/java/project --output_dir ./results --class_name YourClass --package com.example"
echo ""
echo "For help:"
echo "   python run.py --help"
echo "   python lambda_framework.py --help"
echo ""
echo "🎉 LAMBDA Framework installed successfully!" 