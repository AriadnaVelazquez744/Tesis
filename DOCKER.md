# Docker Setup

## Objetivo

Proveer un entorno reproducible con Docker + `uv` para:

- Ejecutar la interfaz tipo chatbot (Streamlit) y llamadas a modelos para inferencia (vía `LLMSTUDIO_BASE_URL`).
- Ejecutar descarga de modelos (`src/base_models/douwnload_models.py`).
- Ejecutar entrenamiento (`src/Vagueness_Judge/training/sft.sh`).

Todo lo ejecutable se hace con scripts `.sh`/`.py`.

## Requisitos (host)

1. Docker (recomendado: Docker Compose v2).
2. Si vas a entrenar/usar GPU con CUDA (opcional; por defecto corre en CPU):
   - GPU NVIDIA compatible con CUDA.
   - Drivers NVIDIA en el host.
   - NVIDIA Container Toolkit en el host (para que Docker habilite `--gpus`).

Nota: en Docker, pedir GPU por CUDA implica stack NVIDIA (CUDA es la plataforma de NVIDIA).
Para activar GPU en Docker sin que sea obligatoria, usá un override opcional:
- CPU (por defecto): `docker compose up -d`
- GPU:
  - `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`

## Archivos clave (ya incluidos)

- `Dockerfile`: construye una imagen con Python 3.12 y un venv creado con `uv` (dentro de `.venv`).
- `docker-compose.yml`: define un servicio `app` reutilizable (web + training).
- `run_web.sh`: arranca la UI (Streamlit).
- `src/base_models/douwnload_models.py`: descarga un modelo base.
- `src/Vagueness_Judge/training/sft.sh`: entrena con tu GPU (llama a `sft.py`).

## Variables de entorno

Crea un archivo `.env` en la raiz del proyecto (`Tesis/.env`) con, al menos:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
LLMSTUDIO_BASE_URL=http://host.docker.internal:1234
```

- `HF_TOKEN`: requerido por `src/base_models/douwnload_models.py`.
- `LLMSTUDIO_BASE_URL`: endpoint del servidor local de modelos (por ejemplo LLMStudio) para que tu chatbot haga inferencia.

## Construir imagen

Desde la carpeta del repo:

```bash
docker compose build
```

## Ejecutar chatbot (Streamlit)

1. Levanta el servicio:

   ```bash
   # CPU (por defecto)
   docker compose up -d

   # GPU (si la necesitas)
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

2. Abre en el navegador: `http://localhost:8501`

El contenedor usa el venv ya creado y `docker-compose` ejecuta `./run_web.sh`.

## Ejecutar descarga de modelos (script .py)

   ```bash
   docker compose run --rm app ./download_models.sh
   ```

## Ejecutar entrenamiento (script .sh)

   ```bash
   docker compose run --rm app ./src/Vagueness_Judge/training/sft.sh
   ```

El script respeta variables como `CUDA_VISIBLE_DEVICES`, `MODEL_DIR_NAME`, etc.

## Persistencia (datos que se guardan)

`docker-compose.yml` monta volumenes para:

- `src/base_models/` (modelos descargados)
- `src/Vagueness_Judge/training_models/` (checkpoints)
- `logs/` (logs de entrenamiento)
- `storage/` (datos persistentes para el chatbot/experimentacion)

## Activar el entorno virtual (venv)

Dentro del contenedor (si necesitas usarlo manualmente):

```bash
docker compose exec app bash
source .venv/bin/activate
```

## uv (opcional, sin Docker)

Si quieres correr localmente con `uv` (sin Docker) en Linux:

1. Crear el venv:

   ```bash
   uv venv --python 3.12 .venv
   ```

2. Instalar dependencias exactamente (usando `uv.lock`):

   ```bash
   uv sync --frozen
   source .venv/bin/activate
   ```

Si agregas/cambias dependencias:

   ```bash
   uv lock
   ```
