FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN mkdir -p /app/tmp

EXPOSE 7860

CMD ["uvicorn", "ingest:app", "--host", "0.0.0.0", "--port", "7860"]






