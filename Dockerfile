FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias mínimas para correr y para compilaciones ocasionales de wheels.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    build-essential \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Instala uv (gestor de dependencias).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
RUN touch /app/.env

# Cache de dependencias: primero copiamos lock/pyproject.
COPY pyproject.toml uv.lock ./

# Crea el entorno virtual del proyecto dentro de la imagen.
RUN uv venv --python 3.12 .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Instala dependencias reproducibles.
RUN uv sync --frozen

# Copia el código y scripts.
COPY main.py ./main.py
COPY src ./src
COPY run_web.sh ./run_web.sh
COPY download_models.sh ./download_models.sh

RUN chmod +x ./run_web.sh ./download_models.sh \
  && mkdir -p ./logs ./storage ./src/base_models ./src/Vagueness_Judge/training_ckpts

EXPOSE 8501
CMD ["./run_web.sh"]

