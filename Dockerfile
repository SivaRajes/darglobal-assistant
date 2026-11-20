FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
COPY scrape.py ./scrape.py
COPY prepare_corpus.py ./prepare_corpus.py
COPY prepare_page_metadata.py ./prepare_page_metadata.py
COPY evaluate_retrieval.py ./evaluate_retrieval.py
EXPOSE 8000
ENV PORT=8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
