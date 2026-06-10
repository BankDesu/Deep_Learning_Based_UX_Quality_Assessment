FROM python:3.11-slim

# System dependencies for OpenCV and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    torch>=2.2.0 torchvision>=0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    timm>=1.0.0 \
    pyyaml>=6.0.1 \
    scipy>=1.12.0 \
    "Pillow>=10.0.0" \
    opencv-python-headless>=4.8.0 \
    numpy \
    matplotlib \
    gradio>=4.0.0 \
    pandas

# Install the uxqa package
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy application code
COPY scripts/ ./scripts/

# Checkpoints are mounted at runtime via volume
# (not baked into image — they are large .pt files)

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/app.py", "--port", "7860"]
