FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY iris_api.py .

CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8080", "iris_api:app"]
