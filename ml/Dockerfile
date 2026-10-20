FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements-serve.txt .
# CPU-only PyTorch keeps the image small (no CUDA download)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-serve.txt
COPY src ./src
COPY service ./service
EXPOSE 8000
# Models are mounted at /app/models (see docker-compose.yml), not baked into the image
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
