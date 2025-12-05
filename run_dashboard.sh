#!/bin/bash

# Building Energy Dashboard Launcher
# Quick script to start the Streamlit dashboard

echo "🏢 Building Energy Performance Dashboard"
echo "========================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "📦 Installing dependencies..."
    pip install -r requirements_streamlit.txt
    echo "✅ Installation complete!"
    echo ""
fi

echo "🚀 Starting dashboard..."
echo "📊 Opening in browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "========================================"
echo ""

# Run streamlit
streamlit run app.py
