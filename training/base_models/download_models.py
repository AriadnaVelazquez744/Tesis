#!/usr/bin/env python3
"""
Descarga modelos base desde Hugging Face a Thesis/training/base_models/
Define MODEL_ID directamente en el script.

Para cambiar modelo, edita la variable MODEL_ID abajo.
Lee HF_TOKEN desde Thesis/.env
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login
import argparse

# =====================================================
# CAMBIA ESTA LÍNEA PARA CADA MODELO QUE QUIERAS
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  
# Otras opciones:
# MODEL_ID = "google/gemma-2-2b-it"           # 2B
# MODEL_ID = "Qwen/Qwen2-7B-Instruct"         # 7B  
# MODEL_ID = "microsoft/DialoGPT-medium"      # 1.5B
# =====================================================

def main():
    # 1. Cargar .env desde Thesis/ (3 niveles arriba)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ .env cargado desde: {env_path}")
    else:
        print(f"❌ .env no encontrado en: {env_path}")
        print("Crea Thesis/.env con: HF_TOKEN=hf_xxxxxxxxxxxxxxxx")
        return
    
    # 2. Verificar HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN no encontrado en .env")
        print("Añade a Thesis/.env: HF_TOKEN=hf_xxxxxxxxxxxxxxxx")
        return
    
    # 3. Configurar paths
    base_models_dir = Path(__file__).parent  # Thesis/training/base_models/
    model_name = MODEL_ID.split("/")[-1]
    model_local_dir = base_models_dir / model_name
    
    print(f"📥 Descargando: {MODEL_ID}")
    print(f"📂 Destino: {model_local_dir}")
    print("-" * 60)
    
    # 4. Login (opcional, snapshot_download maneja token)
    try:
        login(token=hf_token)
        print("✅ Autenticación HF OK")
    except Exception as e:
        print(f"⚠️  Login falló (no crítico): {e}")
    
    # 5. Descargar modelo
    try:
        print("⏳ Descargando archivos del modelo...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=str(base_models_dir / "cache")  # cache local opcional
        )
        print(f"✅ Modelo descargado exitosamente: {model_local_dir}")
        
        # 6. Verificar archivos clave
        required_files = ["config.json"]
        optional_files = ["tokenizer.json", "tokenizer.model", "vocab.json"]
        
        print("\n📋 Verificando archivos descargados:")
        all_files = list(model_local_dir.rglob("*"))
        print(f"   Total archivos: {len(all_files)}")
        
        missing_required = []
        for f in required_files:
            if not (model_local_dir / f).exists():
                missing_required.append(f)
        
        if missing_required:
            print(f"❌ Archivos requeridos faltantes: {missing_required}")
        else:
            print("✅ Todos los archivos requeridos OK")
            
        # Tamaño total
        total_size = sum(f.stat().st_size for f in model_local_dir.rglob("*") if f.is_file())
        print(f"💾 Tamaño total: {total_size / (1024**3):.1f} GB")
        
    except Exception as e:
        print(f"❌ Error descargando {MODEL_ID}:")
        print(f"   {e}")
        return

if __name__ == "__main__":
    main()

