FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p input output

CMD ["python", "main.py"]
