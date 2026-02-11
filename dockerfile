FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
# Install Python 3.10 + minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \ 
    python3.10 \
    python3.10-venv \
    python3-pip \
    libglib2.0-0
    #libsm6 \
    #libxext6 \
    #libxrender1 \
    #&& rm -rf /var/lib/apt/lists/*

# Create and activate venv
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Optional: PyTorch + CUDA 12.6 (uncomment if needed)
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy app + fonts
COPY main.py .
COPY fonts/ ./fonts/

CMD ["python", "main.py"]