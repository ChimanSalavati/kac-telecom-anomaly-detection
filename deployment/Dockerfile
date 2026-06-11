# CPU-only KAC shadow scorer.
# Mirrors the paper's deployment config: the online scorer runs on CPU and makes
# no LLM calls; Chronos-2 residuals and text summaries are computed offline and
# passed in as cached tensors.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0

WORKDIR /app

# Install CPU torch + the model deps, then the serving extras.
COPY requirements.txt /app/requirements.txt
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install transformers numpy \
 && pip install -r /app/deployment/requirements.txt

# Copy the code needed at inference time.
COPY deployment /app/deployment
COPY experiments/_shared /app/experiments/_shared

EXPOSE 8765

# Liveness/readiness are served at /healthz and /readyz (see k8s probes).
CMD ["uvicorn", "deployment.serve_app:app", "--host", "0.0.0.0", "--port", "8765"]
