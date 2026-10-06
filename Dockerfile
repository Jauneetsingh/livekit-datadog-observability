FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py .

# Download the Silero VAD / turn-detector model weights at build time
RUN python agent.py download-files || true

# ddtrace-run enables APM + the OpenAI integration, so the Azure OpenAI calls the
# LiveKit LLM plugin makes (with tool_call arguments) become native LLM Obs spans.
# NOTE: DD_TRACE_OTEL_ENABLED is intentionally NOT set. With it on, Datadog reads
# spans via the OpenTelemetry GenAI convention and expects gen_ai.operation.name,
# which mis-classifies our native LLM Obs spans as generic "workflow".
ENV DD_SERVICE=livekit-loan-agent \
    DD_ENV=demo \
    DD_LLMOBS_ENABLED=1 \
    DD_LLMOBS_ML_APP=livekit-loan-agent \
    DD_LLMOBS_AGENTLESS_ENABLED=0

CMD ["ddtrace-run", "python", "agent.py", "start"]
