#!/usr/bin/env python3
"""
MoGe Model HTTP API Server
Start the FastAPI service via main.py
"""

import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI application
from moge_api import app

def main():
    """Start the FastAPI service"""
    print("Starting MoGe Model HTTP API service...")
    print("Visit http://localhost:8000 to view API documentation")
    
    # Start the application using uvicorn
    uvicorn.run(
        "moge_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload in production environment
    )

if __name__ == "__main__":
    main()