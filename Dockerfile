# ─────────────────────────────────────────────────────────────
#  法小律 · 中国法律法规智能问答 — Docker 镜像
#
#  Build:   docker compose build
#  Run:     docker compose up -d
#  Shell:   docker compose exec law-agent bash
# ─────────────────────────────────────────────────────────────

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="law-agent"
LABEL org.opencontainers.image.description="法小律 — 中国法律法规 RAG + Agent 问答系统"
LABEL org.opencontainers.image.version="0.3.0"

# ── system dependencies ────────────────────────────────────
# build-essential needed for chromadb's native SQLite extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── work directory ─────────────────────────────────────────
WORKDIR /app

# ── Python packages (layer cached for fast rebuilds) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── application code ───────────────────────────────────────
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY assets/    ./assets/
COPY app.py run.py run_agent.py ./

# ── environment ────────────────────────────────────────────
ENV HF_HOME=/app/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONUNBUFFERED=1

# ── Streamlit config ───────────────────────────────────────
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── runtime ────────────────────────────────────────────────
EXPOSE 8501

# Models (BGE-small-zh ~100MB, BGE-reranker-v2-m3 ~2.2GB) download on first
# launch and are persisted via the model_cache Docker volume.
ENTRYPOINT ["streamlit", "run", "app.py"]
