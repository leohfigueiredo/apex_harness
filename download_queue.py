#!/usr/bin/env python3
"""
Automated Model Download Manager for Apex Harness & LM Studio
Downloads the missing GGUF models sequentially to /run/media/leonardo/Windows/AIModels.
Updates progress in JSON and log file continuously.
Synchronizes LM Studio / Apex Harness links upon each completion.
"""

import os
import sys
import time
import json
import subprocess
import traceback

MODELS_QUEUE = [
    # 1. Apex Harness Primary & Lightweight essentials
    ("ukisai/Swift-Qwen3.8-27B-GGUF", "Swift-Qwen3.8-27B-Q4_K_M.gguf", 16.79),
    ("lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", 4.36),
    ("Ma7ee7/Qwen3.8_4B_Distilled_GGUF", "qwen3-4b-thinking-2507.Q8_0.gguf", 3.99),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf", 16.26),
    
    # 2. Reasoning & Coding Specialists
    ("TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF", "glm-4.7-flash-claude-4.5-opus.q4_k_m.gguf", 16.89),
    ("projectj/Instinct-Python-Coder-Gemma4-12B-KimiK3", "Instinct-Python-Coder-Gemma4-12B-KimiK3-Q8_0.gguf", 11.80),
    ("lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF", "gemma-4-26B-A4B-it-QAT-Q4_0.gguf", 13.45),
    ("ibm-granite/granite-4.2-30b-GGUF", "granite-4.2-30b-Q4_K_S.gguf", 15.57),
    ("lmstudio-community/Muse-Glimmer-30B-GGUF", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", 15.61),
    ("lmstudio-community/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf", 16.89),
    
    # 3. High Precision & Extra Models
    ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", 24.94),
    ("lmstudio-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf", 22.83),
    ("unsloth/Qwen3-Coder-Next-GGUF", "Qwen3-Coder-Next-UD-Q3_K_S.gguf", 31.03),
    
    # 4. Large flagship models (downloaded last)
    ("Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF", "qwen38-keep1-Q3KXL.gguf", 60.34),
    ("antirez/deepseek-v4-gguf", "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf", 80.76),
]

BASE_DIR = "/run/media/leonardo/Windows/AIModels"
STATUS_FILE = os.path.join(BASE_DIR, "download_status.json")
LOG_FILE = os.path.join(BASE_DIR, "download_queue.log")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(current_idx, current_model, state, extra=None):
    data = {
        "timestamp": time.time(),
        "human_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(MODELS_QUEUE),
        "current_index": current_idx + 1,
        "current_repo": current_model[0],
        "current_file": current_model[1],
        "estimated_gb": current_model[2],
        "state": state,
    }
    if extra:
        data.update(extra)
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"Erro ao salvar status: {e}")


def sync_lm_studio():
    """Recria symlinks e atualiza LM Studio como no fix_ai_models.sh"""
    log("Sincronizando links com LM Studio...")
    try:
        script = "/home/leonardo/Desktop/fix_ai_models.sh"
        if os.path.exists(script):
            subprocess.run(["bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("Sincronização concluída com sucesso.")
    except Exception as e:
        log(f"Aviso na sincronização: {e}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    log("=== Iniciando Fila Automatizada de Download de Modelos ===")
    log(f"Diretório de Destino: {BASE_DIR}")
    log(f"Total de Modelos na Fila: {len(MODELS_QUEUE)}")

    for idx, (repo, filename, est_gb) in enumerate(MODELS_QUEUE):
        dest_dir = os.path.join(BASE_DIR, repo)
        os.makedirs(dest_dir, exist_ok=True)
        final_file = os.path.join(dest_dir, filename)

        if os.path.exists(final_file) and os.path.getsize(final_file) > 10 * 1024 * 1024:
            size_gb = os.path.getsize(final_file) / (1024**3)
            log(f"[{idx+1}/{len(MODELS_QUEUE)}] Já existe: {repo}/{filename} ({size_gb:.2f} GB). Pulando...")
            write_status(idx, (repo, filename, est_gb), "skipped_already_exists")
            continue

        log(f"[{idx+1}/{len(MODELS_QUEUE)}] BAIXANDO: {repo} -> {filename} (~{est_gb:.1f} GB)...")
        write_status(idx, (repo, filename, est_gb), "downloading")

        start_t = time.time()
        cmd = [
            "hf", "download",
            repo, filename,
            "--local-dir", dest_dir
        ]

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in iter(p.stdout.readline, ''):
                if not line:
                    break
                # Log any relevant progress info
                clean = line.strip()
                if "%" in clean or "error" in clean.lower() or "download" in clean.lower():
                    log(f"  [hf] {clean}")
            p.wait()

            if p.returncode == 0:
                elapsed = time.time() - start_t
                final_sz_gb = os.path.getsize(final_file) / (1024**3) if os.path.exists(final_file) else est_gb
                log(f"[{idx+1}/{len(MODELS_QUEUE)}] ✅ CONCLUÍDO: {filename} ({final_sz_gb:.2f} GB) em {elapsed/60:.1f} min.")
                write_status(idx, (repo, filename, est_gb), "completed", {"elapsed_seconds": elapsed})
                sync_lm_studio()
            else:
                log(f"[{idx+1}/{len(MODELS_QUEUE)}] ❌ Falha no download (código {p.returncode}). Continuando fila...")
                write_status(idx, (repo, filename, est_gb), "failed", {"exit_code": p.returncode})

        except Exception as e:
            log(f"[{idx+1}/{len(MODELS_QUEUE)}] ❌ Erro de execução: {e}")
            traceback.print_exc()

    log("=== Todos os downloads da fila foram processados! ===")
    sync_lm_studio()


if __name__ == "__main__":
    main()
