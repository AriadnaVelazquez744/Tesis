FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_HTTP_TIMEOUT=180
ENV UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    build-essential \
    libgomp1 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
RUN touch /app/.env

COPY pyproject.toml uv.lock ./

RUN uv venv --python 3.12 .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

RUN mkdir -p "${UV_CACHE_DIR}" \
  && uv sync --frozen

RUN uv python install 3.8

RUN uv venv --python 3.8 .venv-amr

RUN . .venv-amr/bin/activate && \
    uv pip install --python .venv-amr/bin/python \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    "numpy<1.24" \
    "torch>=2.1.0" \
    "transformers>=4.35,<4.40" \
    "fairseq<=0.10.2" \
    "penman>=1.1.0" \
    "tqdm>=4.55" \
    "packaging>=20.8" \
    "requests>=2.25" \
    "python-dateutil>=2.8" \
    "ipdb<=0.13.13" \
    "progressbar" \
    "tensorboardX"

RUN . .venv-amr/bin/activate && \
    uv pip install --python .venv-amr/bin/python \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN . .venv-amr/bin/activate && \
    uv pip install --python .venv-amr/bin/python \
    git+https://github.com/IBM/transition-amr-parser.git@v0.5.4

COPY start_services.sh ./start_services.sh
COPY main.py ./main.py
COPY src ./src

RUN chmod +x ./start_services.sh \
  && mkdir -p \
    ./logs \
    ./storage \
    ./src/base_models \
    ./src/Vagueness_Judge/training_models \
    ./src/MIDLM/data \
    ./src/MIDLM/trained_models \
    ./src/MIDLM/experiments \
    ./src/TEXTOIR

EXPOSE 8501
CMD ["/bin/bash", "./start_services.sh"]
