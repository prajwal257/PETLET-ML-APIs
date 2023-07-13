FROM python:3.10.12-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && pip install --upgrade pip && pip install -r requirements.txt
CMD uvicorn app:app --port=${PORT} --host=0.0.0.0