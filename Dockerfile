# ─── Morocco BTP Nowcasting — Docker image ───────────────────────────────────
# Build:   docker build -t btp-nowcasting .
# Run app: docker run -p 8501:8501 btp-nowcasting
# Run tests inside container: docker run --rm btp-nowcasting pytest

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (cached layer) ─────────────────────
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir streamlit>=1.32

# ── Copy project source ───────────────────────────────────────────────────
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/
COPY tests/ ./tests/

# ── Install the package in editable mode ─────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Create output directories ─────────────────────────────────────────────
RUN mkdir -p data/interim data/processed data/vintages data/external docs

# ── Streamlit configuration ───────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────
EXPOSE 8501

# ── Default command: launch Streamlit dashboard ───────────────────────────
CMD ["streamlit", "run", "src/lamiaty/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
