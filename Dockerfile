FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Include only the API entrypoint in the runtime image.
COPY api.py /app/api.py

EXPOSE 8000

CMD ["python", "api.py"]
