FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860


CMD ["sh", "-c", "mkdir -p output && python rank.py --candidates data/candidates.jsonl --out output/final_candidates.csv && streamlit run sandbox/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]