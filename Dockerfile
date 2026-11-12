FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install OpenJDK 17 (required by TorchServe) and curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install compatible PyTorch 2.2.2 CPU, Transformers 4.44.2, NumPy 1.26.4, and TorchServe 0.11.0
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 && \
    pip install numpy==1.26.4 transformers==4.44.2 torchserve==0.11.0 torch-model-archiver==0.11.0

COPY torchserve/ /app/torchserve/
COPY model_store/story_continuation.mar /app/model_store/story_continuation.mar

EXPOSE 8090 8091 8092

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8090/ping || exit 1

CMD ["torchserve", "--start", "--ncs", "--model-store", "/app/model_store", "--models", "story_continuation=story_continuation.mar", "--ts-config", "/app/torchserve/config.properties", "--foreground"]
