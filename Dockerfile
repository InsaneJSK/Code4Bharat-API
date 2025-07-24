FROM python:3.10

# Set working directory to current repo root
WORKDIR /app

# Set custom cache locations to avoid shared /.cache issues
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_home
ENV XDG_CACHE_HOME=/tmp/xdg_cache
ENV TORCH_HOME=/tmp/torch_cache

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model into custom cache directory
RUN python -c "\
import os; \
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy all code (since it's already at root)
COPY . .

# Start FastAPI on HF-compatible port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
