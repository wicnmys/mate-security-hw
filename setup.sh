#!/bin/bash
set -e

echo "========================================="
echo "SQL Query Agent Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    echo "   Please install Python 3.10 or higher"
    exit 1
fi

echo "✓ Python version OK: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "   Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
    else
        # Skip creation if user chose not to recreate
        echo "Using existing virtual environment"
        source .venv/bin/activate
        # Skip to dependency installation
        if [ $? -eq 0 ]; then
            echo "✓ Virtual environment activated"
            echo ""
            echo "Upgrading pip..."
            pip install --upgrade pip -q
            echo "Installing dependencies..."
            pip install -r requirements-dev.txt -q
            echo "✓ Dependencies installed"
            echo ""
            # Continue with rest of script
            SKIP_VENV_CREATION=1
        fi
    fi
fi

if [ -z "$SKIP_VENV_CREATION" ]; then
    # Try using built-in venv first
    echo "Attempting to create virtual environment with venv..."
    if python3 -m venv .venv 2>/dev/null; then
        echo "✓ Virtual environment created with venv"
    else
        echo "⚠️  Built-in venv failed, trying virtualenv..."

        # Check if virtualenv is installed
        if ! command -v virtualenv &> /dev/null; then
            echo "Installing virtualenv..."
            pip3 install virtualenv
        fi

        # Create venv with virtualenv
        virtualenv .venv
        echo "✓ Virtual environment created with virtualenv"
    fi

    echo ""

    # Activate virtual environment
    echo "Activating virtual environment..."
    source .venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip -q

    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements-dev.txt -q

    echo "✓ Dependencies installed"
    echo ""
fi

# Check for API keys
echo "Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ] && [ ! -f ".env" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo ""
    echo "   Create a .env file with your API keys:"
    echo "   echo 'OPENAI_API_KEY=your_key_here' > .env"
    echo ""
    echo "   Or export directly:"
    echo "   export OPENAI_API_KEY='your_key_here'"
    echo ""
elif [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Check for Anthropic API key (needed for Claude)
if [ -z "$ANTHROPIC_API_KEY" ] && [ ! -f ".env" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set (needed for Claude)"
    echo ""
    echo "   Add to .env file:"
    echo "   echo 'ANTHROPIC_API_KEY=your_key_here' >> .env"
    echo ""
elif [ -f ".env" ]; then
    # Check if ANTHROPIC_API_KEY is in .env
    if ! grep -q "ANTHROPIC_API_KEY" .env; then
        echo "⚠️  ANTHROPIC_API_KEY not in .env file"
        echo "   Add it with: echo 'ANTHROPIC_API_KEY=your_key_here' >> .env"
    else
        echo "✓ ANTHROPIC_API_KEY found in .env"
    fi
else
    echo "✓ ANTHROPIC_API_KEY is set"
fi

echo ""

# Verify schema file exists
echo "Checking for schema file..."
if [ -f "schemas/dataset.json" ]; then
    num_tables=$(python3 -c "import json; data = json.load(open('schemas/dataset.json')); print(len(data))")
    echo "✓ Schema file found with $num_tables tables"
else
    echo "❌ Schema file not found: schemas/dataset.json"
    echo "   Please ensure the dataset.json file is in the schemas/ directory"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Set your API keys (if not done already):"
echo "     export OPENAI_API_KEY='your_key_here'"
echo "     export ANTHROPIC_API_KEY='your_key_here'"
echo "     # Or create .env file"
echo ""
echo "  3. Run the agent:"
echo "     python main.py \"Show me high-severity events\""
echo ""
echo "  4. Run tests:"
echo "     pytest tests/ -v"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""
