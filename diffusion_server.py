"""
Apex Harness — DiffusionGemma OpenAI API Bridge (diffusion_server.py)

Exposes an OpenAI-compatible /v1/chat/completions HTTP & SSE API over the
low-level ROCm `llama-diffusion-gemma-visual-server` binary.
Supports real-time token block streaming and live canvas denoising updates.
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List

DIFFUSION_BIN_DEFAULT = "/home/leonardo/.unsloth/llama.cpp/build/bin/llama-diffusion-gemma-visual-server"

class DiffusionWorker:
    def __init__(self, model_path: str, bin_path: str = DIFFUSION_BIN_DEFAULT, ngl: int = 31, idle_timeout: int = 300):
        self.model_path = model_path
        self.bin_path = bin_path
        self.ngl = ngl
        self.idle_timeout = idle_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.is_ready = True
        self.log_file = None
        self.last_used = 0
        self._idle_timer: Optional[threading.Timer] = None
        # Strict lazy-loading: NEVER load model weights into RAM/VRAM during init!

    def _ensure_worker(self):
        """Loads the model into RAM/VRAM only when generation is actually requested."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

        if self.proc and self.proc.poll() is None:
            return

        print(f"📦 Lazy-loading DiffusionGemma weights into ROCm VRAM...")
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
        self.log_file = open("/tmp/diffusion_server_worker.log", "w", buffering=1)

        env = os.environ.copy()
        env["NGL"] = str(self.ngl)

        self.proc = subprocess.Popen(
            [self.bin_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            env=env,
            bufsize=1
        )

        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("READY"):
                print("✓ DiffusionGemma weights loaded and ready.")
                break

    def _schedule_unload(self):
        """Unload model from RAM/VRAM after idle period to keep system memory free."""
        if self._idle_timer:
            self._idle_timer.cancel()
        if self.idle_timeout > 0:
            self._idle_timer = threading.Timer(self.idle_timeout, self.stop)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def generate(self, messages: List[Dict[str, str]], n_blocks: int = 4, seed: int = 42, on_canvas_step=None, on_committed=None) -> str:
        with self.lock:
            self._ensure_worker()
            self.last_used = time.time()

            req_id = uuid.uuid4().hex[:8]
            req_file = f"/tmp/dg_req_{req_id}.json"
            payload = {
                "seed": seed,
                "n_blocks": n_blocks,
                "messages": messages
            }
            with open(req_file, "w") as f:
                json.dump(payload, f)

            self.proc.stdin.write(req_file + "\n")
            self.proc.stdin.flush()

            last_committed = ""

            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("F "):
                    # F <block> <step> <total> <json-string>
                    parts = line.split(" ", 4)
                    if len(parts) >= 5 and on_canvas_step:
                        try:
                            block_idx = int(parts[1])
                            step_idx = int(parts[2])
                            total_steps = int(parts[3])
                            canvas_text = json.loads(parts[4])
                            on_canvas_step(block_idx, step_idx, total_steps, canvas_text)
                        except Exception:
                            pass
                elif line.startswith("C "):
                    # C <block> <json-string>
                    parts = line.split(" ", 2)
                    if len(parts) >= 3:
                        try:
                            block_idx = int(parts[1])
                            committed_text = json.loads(parts[2])
                            delta = committed_text[len(last_committed):]
                            last_committed = committed_text
                            if on_committed and delta:
                                on_committed(block_idx, delta)
                        except Exception:
                            pass
                elif line.startswith("DONE"):
                    break
                elif line.startswith("ERR "):
                    raise RuntimeError(line)

            try:
                os.remove(req_file)
            except Exception:
                pass

            self._schedule_unload()
            return last_committed

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                print("🛑 Unloading DiffusionGemma from ROCm VRAM (idle timeout or stop)...")
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
                self.proc = None
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
                self.log_file = None

_WORKER: Optional[DiffusionWorker] = None

class DiffusionHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/health", "/api/ping"):
            if _WORKER and _WORKER.is_ready:
                self._send_json({"status": "ok", "ready": True})
            else:
                self._send_json({"status": "starting", "ready": False}, status=503)
        elif path in ("/v1/models", "/api/models"):
            model_id = os.path.basename(_WORKER.model_path) if _WORKER else "diffusiongemma"
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "unsloth",
                        "permission": []
                    }
                ]
            })
        elif path == "/props":
            model_path = _WORKER.model_path if _WORKER else ""
            self._send_json({
                "model_path": model_path,
                "total_slots": 1
            })
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path in ("/v1/chat/completions", "/api/chat"):
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                req = json.loads(post_body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=400)
                return

            messages = req.get("messages", [])
            stream = bool(req.get("stream", False))
            n_blocks = req.get("n_blocks", 4)
            seed = req.get("seed", int(time.time()) % 100000)

            if not _WORKER:
                self._send_json({"error": "Worker not initialized"}, status=500)
                return

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                model_name = os.path.basename(_WORKER.model_path)

                def on_chunk(b_idx, delta_text):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                try:
                    _WORKER.generate(messages, n_blocks=n_blocks, seed=seed, on_committed=on_chunk)
                    done_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception as e:
                    err_msg = json.dumps({"error": str(e)})
                    self.wfile.write(f"data: {err_msg}\n\n".encode("utf-8"))
            else:
                try:
                    full_text = _WORKER.generate(messages, n_blocks=n_blocks, seed=seed)
                    res = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": os.path.basename(_WORKER.model_path),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": full_text},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    self._send_json(res)
                except Exception as e:
                    self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Not Found")

def main():
    global _WORKER
    parser = argparse.ArgumentParser(description="DiffusionGemma OpenAI API Server Bridge")
    parser.add_argument("--model", required=True, help="Path to DiffusionGemma GGUF model")
    parser.add_argument("--bin", default=DIFFUSION_BIN_DEFAULT, help="Path to llama-diffusion-gemma-visual-server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--ngl", type=int, default=31, help="GPU layers offload (default: 31)")
    args = parser.parse_args()

    print(f"🚀 Initializing DiffusionGemma ROCm Worker for {args.model}...")
    _WORKER = DiffusionWorker(args.model, bin_path=args.bin, ngl=args.ngl)

    server = ThreadingHTTPServer((args.host, args.port), DiffusionHTTPHandler)
    print(f"✓ DiffusionGemma OpenAI API ready at http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _WORKER:
            _WORKER.stop()
        server.server_close()

if __name__ == "__main__":
    main()
