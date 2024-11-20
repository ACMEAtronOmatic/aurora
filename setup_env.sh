#!/bin/bash

venv_name="a_env"

# Function to check for existing virtual environment
check_venv() {
    # Check common virtual environment directory names
    for venv_dir in "$venv_name"; do
        # Check if directory exists and contains key virtualenv files
        if [ -d "$venv_dir" ] && [ -f "$venv_dir/pyvenv.cfg" ]; then
            echo "Found virtual environment in $venv_dir"
            return 0
        fi
    done
    
    echo "No virtual environment found in current directory"
    return 1
}


if ! check_venv; then
    echo "Creating virtual environment '$venv_name'..."
    python3 -m venv "$venv_name"
    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully"
    else
        echo "Failed to create virtual environment"
        exit 1
    fi
else 
    echo "Virtual environment '$venv_name' already exists"
fi

# activate virtual env
source $venv_name/bin/activate

# update and upgrade pip
pip install --upgrade pip

# install myradar specific requirements from requirements.txt
pip install -r requirements.txt

# install aurora dev requirements from pyproject.toml
pip install -e ".[dev]"

echo ""
echo "Environment setup complete"
echo "Please run 'source $venv_name/bin/activate' to activate the virtual environment."