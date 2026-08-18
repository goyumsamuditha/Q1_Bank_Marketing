# Dockerfile
# ----------
# Only needed for the OPTIONAL secondary serving path (a containerised
# FastAPI endpoint behind Azure ML / Azure Container Apps). The PRIMARY
# real-time serving path is Databricks Model Serving, which needs no
# container at all - see the plan's §6.4.
#
# Build and push, versioned by Git commit SHA:
#   az acr build --registry <yourACR> --image bank-term-model:$(git rev-parse --short HEAD) .

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (before copying code) so Docker's layer
# cache is reused on every rebuild where requirements.txt hasn't changed -
# this alone can save minutes per CI run.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the trained model artefacts.
COPY src/ src/
COPY models/ models/

ENV MODEL_PATH=/app/models/final_pipeline.joblib
ENV THRESHOLD_PATH=/app/models/decision_threshold.txt
ENV SEASONAL_LOOKUP_PATH=/app/models/seasonal_conversion_prior.csv
ENV FREQ_ENCODERS_PATH=/app/models/frequency_encoders.pkl

EXPOSE 8000

# A basic healthcheck so orchestrators (Azure Container Apps, Kubernetes)
# know when the container is actually ready to serve, not just "started".
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
