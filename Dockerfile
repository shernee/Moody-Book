FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p output/audio

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
