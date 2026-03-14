FROM python:3.12-slim

WORKDIR /app

# System deps for headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create workspace directory
RUN mkdir -p /app/workspace

# Cloud Run uses PORT env variable (default 8080)
ENV PORT=8080
ENV WORKSPACE=/app/workspace
# Disable self-modification in cloud (ephemeral containers)
ENV SYNAPSE_CLOUD_MODE=1

EXPOSE ${PORT}

# Single worker required for SocketIO; eventlet for async WebSocket support
# Cloud Run injects PORT env var
# --keep-alive 65: Cloud Run LB timeout is 60s, keep server side longer
# --access-logfile -: log all requests to stdout for Cloud Logging
# --error-logfile -: log errors to stderr
# --log-level warning: reduce noise, only warnings+errors
CMD exec gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --bind 0.0.0.0:${PORT} \
    --timeout 300 \
    --graceful-timeout 30 \
    --keep-alive 65 \
    --access-logfile - \
    --error-logfile - \
    --log-level warning \
    "agent_ui:app"
