#!/bin/bash

# Make sure the required packages are installed
pip install streamlit pandas numpy matplotlib plotly statsmodels pytrends prophet fredapi yfinance scikit-learn

# Check if the output directory already exists
if [ ! -d "../output" ]; then
  echo "Creating output directory..."
  mkdir -p ../output/Cohorts_BTC
  mkdir -p ../output/Dynamic_Plot
fi

# Check if the functions directory exists
if [ ! -d "../functions" ]; then
  echo "ERROR: The functions directory does not exist in the repository root."
  echo "Make sure the repository is correctly structured and that the dashboard"
  echo "is placed in the 'streamlit' folder within the repository root."
  exit 1
fi

# Run the Streamlit app
streamlit run app.py