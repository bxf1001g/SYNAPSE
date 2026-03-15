FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentinel.py .

ENV PORT=8080
EXPOSE 8080

CMD gunicorn sentinel:app --bind 0.0.0.0:$PORT --worker-class gthread --threads 2 --workers 1 --timeout 300
