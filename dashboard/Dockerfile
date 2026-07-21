# Stage 1 (builder): install the Python dependencies into a virtual
# environment. Doing this in its own stage keeps pip's download cache and
# any build leftovers out of the image we actually ship.
FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 (runtime): the image we actually run. It copies the finished
# virtual environment from the builder plus the app code, and nothing
# else, so it stays lean.
FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app.py .
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
