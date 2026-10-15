# Alerts API image (serving only: reads gold Delta tables, no Spark/JVM needed at runtime)
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]"
ENV PAYLAKE_LAKE_ROOT=/data/lake
EXPOSE 8000
USER nobody
CMD ["uvicorn", "paylake.app:app", "--host", "0.0.0.0", "--port", "8000"]
