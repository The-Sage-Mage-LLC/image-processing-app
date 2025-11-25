# Dockerfile for Image Processing Application
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILD_VERSION=1.0.0

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY src ./src

# Install build dependencies and build package
RUN pip install --no-cache-dir build wheel && \
    python -m build --wheel

# Production stage
FROM python:3.11-slim

# Set metadata
LABEL org.opencontainers.image.title="Image Processing Application"
LABEL org.opencontainers.image.description="Enterprise image processing with AI analysis"
LABEL org.opencontainers.image.version="${BUILD_VERSION}"
LABEL org.opencontainers.image.authors="The-Sage-Mage <contact@thesagemage.com>"
LABEL org.opencontainers.image.url="https://github.com/The-Sage-Mage-LLC/image-processing-app"
LABEL org.opencontainers.image.source="https://github.com/The-Sage-Mage-LLC/image-processing-app"
LABEL org.opencontainers.image.vendor="The Sage Mage LLC"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r imgproc && useradd -r -g imgproc -s /bin/bash imgproc

# Create application directories
RUN mkdir -p /app /data/input /data/output /data/admin /logs && \
    chown -R imgproc:imgproc /app /data /logs

# Set work directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir --no-deps *.whl[web] && \
    rm -f *.whl

# Copy configuration files
COPY config/ ./config/
COPY api_launcher.py ./

# Set proper permissions
RUN chown -R imgproc:imgproc /app

# Switch to non-root user
USER imgproc

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV IMGPROC_LOG_LEVEL=INFO
ENV IMGPROC_MAX_PARALLEL_WORKERS=4
ENV IMGPROC_ENABLE_GPU=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["python", "api_launcher.py", "--host", "0.0.0.0", "--port", "8000", "--prod"]

# Alternative entry points
# CMD ["python", "-m", "src.cli.main", "--help"]  # CLI mode
# CMD ["python", "gui_launcher.py"]  # GUI mode (requires X11 forwarding)