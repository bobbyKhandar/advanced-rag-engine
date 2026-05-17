FROM python:3.11-slim

RUN groupadd -r app && useradd -r -g app -d /app -s /bin/false app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=app:app . .

USER app

CMD ["python", "-m", "src.main"]
