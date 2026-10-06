"""LiveKit voice agent (loan bot) instrumented for Datadog LLM Observability.

WHY A BRIDGE: LiveKit's own OpenTelemetry telemetry does not reach Datadog in
this stack (the ddtrace OTEL-provider bridge drops the spans, the Agent's OTLP
collector won't bind, and ddtrace can't auto-instrument the openai v2 SDK that
LiveKit requires). So we instrument the agent's own hooks with the ddtrace
`LLMObs` SDK — which IS enabled and ships over the proven APM intake (8126).
Result: every LLM turn and every tool call shows up in Agent Observability,
grouped per call by session_id (the LiveKit room).

Run:  ddtrace-run python agent.py start
"""
import logging
import os

from livekit.agents import (Agent, AgentSession, JobContext, JobExecutorType,
                            RunContext, WorkerOptions, cli, function_tool)
from livekit.plugins import azure, openai, silero

from ddtrace import tracer
from ddtrace.llmobs import LLMObs

logger = logging.getLogger("livekit-loan-agent")
ML_APP = os.environ.get("DD_LLMOBS_ML_APP", "livekit-loan-agent")
DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Enable LLM Obs at import time in the MAIN (ddtrace-instrumented) process. Jobs
# run as THREADS here (see WorkerOptions below), so the bridge spans are created
# and flushed by the same tracer that already ships to the Agent on :8126.
try:
    LLMObs.enable(ml_app=ML_APP)
except Exception:
    pass

SYSTEM_PROMPT = (
    "You are a friendly, concise personal-loan sales agent for Contoso Bank on a "
    "live phone call. Ask the caller's loan purpose and approximate annual income, "
    "then call check_loan_eligibility and present the indicative offer: an APR at or "
    "above 6.5% and an amount up to ~30% of annual income (max $50,000). When the "
    "caller agrees, call create_loan_application. If they're upset or ask for a "
    "person, call transfer_to_human. Never read back SSN or card numbers. Keep "
    "replies to one or two short, natural spoken sentences."
)
MAX_LOAN, FLOOR_APR = 50000, 6.5


def _last_user_text(chat_ctx):
    """Best-effort extract the latest user utterance from a LiveKit ChatContext."""
    try:
        items = getattr(chat_ctx, "items", None) or getattr(chat_ctx, "messages", []) or []
        for m in reversed(items):
            if getattr(m, "role", None) == "user":
                return getattr(m, "text_content", None) or str(getattr(m, "content", ""))[:400]
    except Exception:
        pass
    return "<user turn>"


class LoanAgent(Agent):
    def __init__(self, session_id: str, root=None):
        super().__init__(instructions=SYSTEM_PROMPT)
        self._sid = session_id
        self._root = root

    def _anchor(self):
        # Re-attach to the held-open call root so this span nests UNDER it
        # (LiveKit runs each node/tool in its own task, which resets the active
        # span; without this every span becomes its own tiny trace).
        if self._root is not None:
            try:
                tracer.context_provider.activate(self._root)
            except Exception:
                pass

    # ---- LLM turn: DON'T create our own span — ddtrace's openai integration
    # already emits a rich `AzureOpenAI.createChatCompletion` llm span (real
    # messages, tokens, and tool_calls). We only re-anchor to the call root so
    # that auto span (and the tool spans below) nest into the ONE call trace. ---
    async def llm_node(self, *args, **kwargs):
        self._anchor()
        async for chunk in super().llm_node(*args, **kwargs):
            yield chunk

    # ---- Tools: each call becomes an LLM Obs `tool` span (visible tool calls) --
    @function_tool
    async def check_loan_eligibility(self, context: RunContext,
                                     annual_income: float, requested_amount: float = 0.0):
        """Check whether the caller qualifies and return an indicative APR and amount.
        annual_income: caller's approximate yearly income in USD.
        requested_amount: amount the caller wants to borrow in USD (optional)."""
        self._anchor()
        with LLMObs.tool(name="check_loan_eligibility", session_id=self._sid) as span:
            LLMObs.annotate(span=span, input_data={"annual_income": annual_income,
                                                    "requested_amount": requested_amount},
                            tags={"framework": "livekit"})
            approved = annual_income >= 30000
            base = min(round(annual_income * 0.3, -2), MAX_LOAN)
            amount = min(requested_amount, base) if requested_amount else base
            apr = FLOOR_APR if annual_income >= 60000 else 8.5
            result = {"approved": approved, "apr": apr if approved else None,
                      "amount": amount if approved else 0}
            LLMObs.annotate(span=span, output_data=result)
            return result

    @function_tool
    async def create_loan_application(self, context: RunContext,
                                      amount: float, apr: float, purpose: str = "personal"):
        """Submit the loan application once the caller agrees to the offer."""
        self._anchor()
        with LLMObs.tool(name="create_loan_application", session_id=self._sid) as span:
            LLMObs.annotate(span=span, input_data={"amount": amount, "apr": apr, "purpose": purpose},
                            tags={"framework": "livekit"})
            result = {"status": "submitted", "amount": amount, "apr": apr,
                      "purpose": purpose, "reference": "LN-" + os.urandom(3).hex().upper()}
            LLMObs.annotate(span=span, output_data=result)
            return result

    @function_tool
    async def transfer_to_human(self, context: RunContext, reason: str = "caller request"):
        """Escalate the call to a human loan officer."""
        self._anchor()
        with LLMObs.tool(name="transfer_to_human", session_id=self._sid) as span:
            LLMObs.annotate(span=span, input_data={"reason": reason},
                            output_data={"status": "transferring"}, tags={"framework": "livekit"})
            return {"status": "transferring", "reason": reason}


async def entrypoint(ctx: JobContext):
    try:
        LLMObs.enable(ml_app=ML_APP)   # idempotent; ddtrace-run already enables it
    except Exception:
        pass

    await ctx.connect()
    session_id = ctx.room.name        # one call == one LLM Obs session + one trace
    # held-open root so the whole call is ONE agent trace (turns/tools/tts nest in)
    root = LLMObs.workflow(name="voice_agent_call", session_id=session_id)
    LLMObs.annotate(span=root, input_data="<voice call>",
                    tags={"framework": "livekit", "channel": "voice"})
    logger.info("LLM Obs bridge active — ml_app=%s session_id=%s", ML_APP, session_id)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=azure.STT(),                       # AZURE_SPEECH_KEY / AZURE_SPEECH_REGION
        tts=azure.TTS(),                       # AZURE_SPEECH_KEY / AZURE_SPEECH_REGION
        llm=openai.LLM.with_azure(             # AZURE_OPENAI_*
            azure_deployment=DEPLOY,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        ),
    )

    async def _shutdown():
        try:
            LLMObs.annotate(span=root, output_data="call ended")
            root.finish()          # closes the single call trace
        except Exception:
            pass
        try:
            LLMObs.flush()
        except Exception:
            pass
    ctx.add_shutdown_callback(_shutdown)

    await session.start(agent=LoanAgent(session_id, root), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the caller warmly and ask what they'd like a loan for "
                     "and their rough annual income.")


if __name__ == "__main__":
    # THREAD executor keeps jobs in the main process so ddtrace/LLM Obs traces and
    # flushes them (PROCESS executor forks a subprocess whose spans don't export).
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,
                              job_executor_type=JobExecutorType.THREAD))
