# Use official Python runtime as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install uv
RUN pip install uv

# Use uv to install project dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Start service
CMD ["uv", "run", "python", "main.py"]