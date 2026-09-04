FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_RUNTIME_DIR=/app/runtime

WORKDIR /app

COPY requirements.txt requirements-langchain.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/runtime \
    && chown -R agent:agent /app

USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "lesson_08_streaming_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
