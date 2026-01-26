# Finance Agents - AI-powered Oslo Stock Exchange analysis
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MPLBACKEND=Agg

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (--progress-bar off avoids threading issues in Docker)
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy application code
COPY . .

# Create data directory for SQLite and reports
RUN mkdir -p /data/reports

# Set default environment variables
ENV DB_PATH=/data/finance_agents.db \
    REPORTS_DIR=/data/reports

# Make analyze.py executable
RUN chmod +x analyze.py

# Default entrypoint
ENTRYPOINT ["python", "analyze.py"]

# Default command (shows help)
CMD ["--help"]
