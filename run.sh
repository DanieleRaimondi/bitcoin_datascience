#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p data pages

# Check if Python dependencies are installed
echo "Checking dependencies..."
pip install -q streamlit pandas numpy matplotlib statsmodels pillow requests

# Make sure functions directory is accessible
if [ ! -d "functions" ]; then
    echo "Warning: 'functions' directory not found. Creating symbolic link..."
    
    # Try to find the functions directory in the project
    if [ -d "../functions" ]; then
        ln -s ../functions functions
    elif [ -d "/Users/danieleraimondi/bitcoin_datascience/functions" ]; then
        ln -s /Users/danieleraimondi/bitcoin_datascience/functions functions
    else
        echo "Error: Could not locate functions directory. Please ensure it exists before running."
        exit 1
    fi
fi

# Check if required data files exist
if [ ! -f "data/thermomodel.csv" ]; then
    echo "Note: ThermoModel data file not found. It will be generated on first run."
fi

# Set environment variables for Streamlit
export PYTHONPATH=$PYTHONPATH:./functions

# Run the Streamlit app
echo "Starting Bitcoin Analysis Dashboard..."
streamlit run Home.py --server.port=8501 --server.headless=false