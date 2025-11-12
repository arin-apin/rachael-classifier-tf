#!/bin/bash

# Script to start the container with GPU support

echo "🔍 Detecting system configuration..."

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected!"

    # Check if nvidia-docker runtime is available
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "✅ NVIDIA Docker runtime detected"
        export DOCKER_RUNTIME=nvidia
        export MEMORY_LIMIT=12G
        export MEMORY_RESERVATION=6G
    else
        echo "⚠️  NVIDIA GPU found but nvidia-docker runtime not available"
        echo "   Installing nvidia-container-toolkit is recommended for GPU support"
        echo "   Falling back to CPU mode..."
        export DOCKER_RUNTIME=runc
        export MEMORY_LIMIT=8G
        export MEMORY_RESERVATION=4G
    fi
else
    echo "ℹ️  No NVIDIA GPU detected, using CPU mode"
    export DOCKER_RUNTIME=runc
    export MEMORY_LIMIT=8G
    export MEMORY_RESERVATION=4G
fi

# Get user and group ID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo ""
echo "🚀 Starting Docker container with configuration:"
echo "   - Runtime: $DOCKER_RUNTIME"
echo "   - Memory Limit: $MEMORY_LIMIT"
echo "   - Memory Reservation: $MEMORY_RESERVATION"
echo "   - User ID: $USER_ID"
echo "   - Group ID: $GROUP_ID"
echo ""

# Stop and remove existing container
echo "🛑 Stopping existing container (if any)..."
docker compose down

# Build and start with new configuration
echo "🔨 Building and starting container..."
docker compose up --build "$@"
