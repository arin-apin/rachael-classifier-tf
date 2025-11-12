# Multi-stage build for GPU/CPU compatibility
# Uses NVIDIA TensorFlow image as base but works on CPU if GPU is not available
FROM nvcr.io/nvidia/tensorflow:22.09-tf2-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# GPU-related environment variables (will be ignored if no GPU present)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# TensorFlow is already included in base image with GPU support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user that matches the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash appuser

# Copy application code
COPY . .

# Create directories for models, data and logs with proper ownership
RUN mkdir -p /workspace/models /workspace/data /workspace/logs && \
    chown -R appuser:appuser /workspace

# Set permissions
RUN chmod +x /workspace/training_app.py

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "training_app.py"]
