FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app.py .

COPY src/models/resnet/resnet50_politician.pth      /app/src/models/resnet/resnet50_politician.pth
COPY src/models/resnet/classes.json                 /app/src/models/resnet/classes.json
COPY src/models/efficientnet/efficientnet_politician.pth /app/src/models/efficientnet/efficientnet_politician.pth
COPY src/models/efficientnet/classes.json           /app/src/models/efficientnet/classes.json

COPY frontend/ /app/frontend/

ENV TOP_N=5 PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]