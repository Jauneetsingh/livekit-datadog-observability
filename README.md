# LiveKit Voice Agent → Datadog Agent Observability

A real [LiveKit](https://livekit.io) Cloud voice agent — a loan-sales bot that
answers a live call, talks to the caller, and makes **function/tool calls** — wired
up so that **everything happening inside LiveKit is visible in Datadog LLM
Observability**: every LLM turn (with tokens and tool-call arguments) and every tool
execution, grouped as **one trace per phone call**.

This repo exists to answer a common question: *"My LiveKit agent runs, but nothing
shows up in Datadog — no LLM turns, no tool calls. Why, and how do I fix it?"*

## Why LiveKit is invisible by default (and how this fixes it)

| Problem | Fix (in this repo) |
|---|---|
| LiveKit forks each call into a **subprocess**, whose tracer never flushes | `JobExecutorType.THREAD` — run jobs in the main, instrumented process |
| Each LiveKit node/tool runs in its own task, so spans scatter into tiny separate traces | A **held-open root** workflow span per call + `tracer.context_provider.activate(root)` to re-anchor every span under it |
| LLM turns and tool calls aren't captured | `ddtrace`'s `openai` integration auto-captures the Azure OpenAI call; tools are wrapped in `LLMObs.tool(...)` spans |
| Spans show **"Missing operation name (`gen_ai.operation.name`)"** and get mis-classified | **Do NOT set `DD_TRACE_OTEL_ENABLED`** — it makes Datadog read spans via the OpenTelemetry GenAI convention instead of the native LLM Obs one |

Full write-up with an annotated example trace: [`docs/livekit-datadog-agent-observability.pdf`](docs/livekit-datadog-agent-observability.pdf)
and [`docs/livekit-datadog-guide.html`](docs/livekit-datadog-guide.html).

## Prerequisites

- **Docker** (Docker Desktop / Rancher Desktop) with Compose
- A **Datadog** account + [API key](https://app.datadoghq.com/organization-settings/api-keys)
- A **LiveKit Cloud** project — URL + API key/secret from *Project Settings → Keys*
- **Azure OpenAI** (a `gpt-4o` deployment) and **Azure Speech** (STT + TTS) resources

## Setup

```bash
cp .env.example .env
# open .env and fill in your own keys — this file is gitignored, never commit it
```

## Run

```bash
docker compose up -d --build
docker compose logs -f livekit-agent     # wait for "registered worker"
```

The worker now waits for a call. Start one from the hosted
**[LiveKit Agents Playground](https://agents-playground.livekit.io)** — connect it to
the **same** LiveKit project as your `.env`, then just talk: *"I'd like a loan for a
car, I make about 80,000 a year."* The agent replies, checks eligibility, and (if you
agree) creates an application — each of those is a tool call.

## See it in Datadog

1. Go to **LLM Observability** → set **ml_app** to `livekit-loan-agent`.
2. Open the latest trace. You'll see **one `voice_agent_call` trace per call**, with:
   - `AzureOpenAI.createChatCompletion` **llm** spans — real messages, token counts, and `tool_calls`
   - `check_loan_eligibility` / `create_loan_application` / `transfer_to_human` **tool** spans — arguments in, result out
   - all grouped by **session_id** (the LiveKit room name)

## How it works (the code)

Everything is in [`agent.py`](agent.py). The key pieces:

```python
# 1) Enable LLM Obs in the main process at import time
LLMObs.enable(ml_app=ML_APP)

# 2) Run jobs as THREADS so spans flush from the instrumented process
cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,
                          job_executor_type=JobExecutorType.THREAD))

# 3) One held-open root span per call == one trace, grouped by room name
root = LLMObs.workflow(name="voice_agent_call", session_id=ctx.room.name)

# 4) Re-anchor to that root before each node/tool so spans nest under it
tracer.context_provider.activate(self._root)

# 5) Each tool is its own LLM Obs span (visible arguments + result)
with LLMObs.tool(name="check_loan_eligibility", session_id=self._sid) as span:
    LLMObs.annotate(span=span, input_data=..., output_data=...)
```

And in the [`Dockerfile`](Dockerfile): the process runs under `ddtrace-run`
(which enables the `openai` integration), and **`DD_TRACE_OTEL_ENABLED` is left
unset on purpose** — that single setting is the usual cause of the
"Missing operation name" misclassification.

## Swapping providers

STT/TTS use the Azure Speech plugin and the LLM uses Azure OpenAI. To use Deepgram,
Cartesia, OpenAI, etc., change the plugins in `AgentSession(...)` in `agent.py` and
the matching env vars — the Datadog instrumentation is unaffected.

## Security

- `.env` holds real secrets and is **gitignored** — never commit it. Use
  `.env.example` (placeholders only) as the template.
- If a secret is ever exposed, rotate it in the provider's console.
