FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV TFHUB_CACHE_DIR=/app/tfhub_cache
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py",  "--server.address=0.0.0.0",  "--server.port=8501",  "--server.enableCORS=false",  "--server.enableXsrfProtection=false", "--server.headless=true", "--server.runOnSave=false"]
