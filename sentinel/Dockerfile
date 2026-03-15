FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentinel.py .

ENV PORT=8081
EXPOSE 8081

CMD ["gunicorn", "sentinel:app", \
     "--bind", "0.0.0.0:8081", \
     "--worker-class", "gthread", \
     "--threads", "2", \
     "--workers", "1", \
     "--timeout", "300"]
