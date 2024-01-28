#!/bin/bash

# Create Python virtual environment
python3 -m venv env
# Activate the virtual environment
source ./env/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Run the Streamlit application
streamlit run LLM_streamlit.py
