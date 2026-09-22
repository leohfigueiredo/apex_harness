#!/usr/bin/env python3
"""
Apex Harness — Monitor ao Vivo (monitor.py)
============================================
Abre uma janela de terminal que mostra em tempo real:

  • Estado do servidor (online / gerando / idle)
  • Tokens no contexto e % de uso
  • Velocidade de geração (t/s, suavizada)
  • Cache KV (% do prompt reutilizado)
  • Tokens da sessão (lidos + escritos)
  • Último turno (prompt + completion tokens)
  • RAM do sistema

Uso:
    python -m apex_harness.monitor
    python -m apex_harness.monitor --port 7860
    python -m apex_harness.monitor --url http://127.0.0.1:7860

Tecla Ctrl-C para sair.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
import json
from typing import Any, Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ─────────────────────────────────────────────────────────────────────────────
# Suavização EMA idêntica à do cli.py para números estáveis no display.
# ─────────────────────────────────────────────────────────────────────────────
_SMOOTH: Dict[str, float] = {}


def _ema(key: str, value: float, alpha: float = 0.25) -> float:
    v = float(value or 0.0)
    if key not in _SMOOTH:
        _SMOOTH[key] = v
        return v
    _SMOOTH[key] = _SMOOTH[key] + alpha * (v - _SMOOTH[key])
    return _SMOOTH[key]


def _fetch(url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _n(v: Any, decimals: int = 0) -> str:
    try:
        f = float(v or 0)
        if decimals:
            return f"{f:,.{decimals}f}".replace(",", " ")
        return f"{int(f):,}".replace(",", " ")
    except Exception:
        return "—"


def _bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _state_color(status: str, decode_tps: float) -> str:
    if status == "ready" and decode_tps > 0.5:
        return "bold green"
    if status == "ready":
        return "bold cyan"
    if status == "loading":
        return "bold yellow"
    return "bold red"


def _state_label(status: str, decode_tps: float, elapsed_gen: float) -> str:
    if status == "ready" and decode_tps > 0.5:
        return f"⚡ GERANDO  ({elapsed_gen:.0f}s)"
    if status == "ready":
        return "✓  IDLE — aguardando input"
    if status == "loading":
        return "⏳ Carregando modelo..."
    return "✗  OFFLINE"


_gen_start: float = 0.0
_last_tps: float = 0.0


def build_dashboard(data: Optional[Dict[str, Any]], server_url: str) -> Panel:
    global _gen_start, _last_tps

    if data is None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column()
        grid.add_row(Text("✗  Servidor não responde — aguardando...", style="bold red"))
        grid.add_row(Text(f"   URL: {server_url}", style="dim"))
        return Panel(grid, title="[bold]⚡ Apex Harness Monitor[/bold]",
                     border_style="red", padding=(1, 2))

    status      = str(data.get("status", "offline"))
    model       = str(data.get("model", "—"))
    ctx_tokens  = int(data.get("context_tokens") or 0)
    max_ctx     = int(data.get("max_context") or 65536)
    ctx_pct     = float(data.get("context_pct") or 0)
    comp_tokens = int(data.get("completion_tokens") or 0)
    cached_tok  = int(data.get("cached_tokens") or 0)
    ram_pct     = float(data.get("ram_pct") or 0)
    decode_raw  = float(data.get("decode_tps") or 0)
    prefill_raw = float(data.get("prefill_tps") or 0)
    sess_prompt = int(data.get("session_prompt_tokens") or 0)
    sess_comp   = int(data.get("session_completion_tokens") or 0)
    msgs        = int(data.get("messages_count") or 0)

    decode_tps  = _ema("decode_tps", decode_raw)
    prefill_tps = _ema("prefill_tps", prefill_raw)

    if decode_tps > 0.3 and _last_tps <= 0.3:
        _gen_start = time.time()
    if decode_tps <= 0.3:
        _gen_start = 0.0
    _last_tps = decode_tps

    elapsed_gen = (time.time() - _gen_start) if _gen_start else 0.0
    cache_pct = round(cached_tok / ctx_tokens * 100) if ctx_tokens > 0 and cached_tok > 0 else 0

    state_label = _state_label(status, decode_tps, elapsed_gen)
    state_color = _state_color(status, decode_tps)

    grid = Table.grid(padding=(0, 1))
    grid.add_column(min_width=26, style="dim")
    grid.add_column()

    def row(label: str, value: Any, value_style: str = ""):
        grid.add_row(
            Text(label, style="dim"),
            Text(str(value), style=value_style) if value_style else Text(str(value))
        )

    def divider():
        grid.add_row(Text(""), Text("─" * 44, style="dim"))

    row("Estado", state_label, state_color)
    row("Modelo", model.split("/")[-1] if "/" in model else model, "bold white")
    row("Mensagens no histórico", str(msgs))
    divider()

    ctx_bar = _bar(ctx_pct, 24)
    row("Contexto", f"{ctx_bar}  {_n(ctx_tokens)} / {_n(max_ctx)} tok  {ctx_pct:.1f}%")
    divider()

    if decode_tps > 0.3:
        row("Decode (geração)",   f"{decode_tps:.1f} t/s", "bold green")
    else:
        row("Decode (geração)",   "— t/s  (idle)", "dim")

    if prefill_tps > 0.5:
        row("Prefill (prompt)",   f"{prefill_tps:.0f} t/s", "green")
    else:
        row("Prefill (prompt)",   "— t/s", "dim")

    row("Último turno",
        f"↑ {_n(ctx_tokens)} prompt   ↓ {_n(comp_tokens)} completion")

    if cache_pct > 0:
        row("Cache KV", f"{_bar(cache_pct, 24)}  {cache_pct}% reutilizado", "cyan")
    else:
        row("Cache KV", "♻  sem cache (primeiro turno)", "dim")

    divider()

    row("Sessão — lido",    f"{_n(sess_prompt)} tokens")
    row("Sessão — escrito", f"{_n(sess_comp)} tokens")
    row("Sessão — total",   f"{_n(sess_prompt + sess_comp)} tokens", "bold white")
    divider()

    ram_color = "red" if ram_pct > 85 else "yellow" if ram_pct > 70 else "green"
    row("RAM sistema", f"{_bar(ram_pct, 24)}  {ram_pct:.0f}%", ram_color)

    border = "green" if decode_tps > 0.3 else "cyan" if status == "ready" else "red"
    ts = time.strftime("%H:%M:%S")
    title = f"[bold]⚡ Apex Harness Monitor[/bold]  [dim]{ts}[/dim]"
    return Panel(grid, title=title, border_style=border, padding=(1, 2))


def run(server_url: str = "http://127.0.0.1:7860", interval: float = 1.0):
    console = Console()
    status_url = server_url.rstrip("/") + "/api/status"

    console.print(f"\n[bold cyan]⚡ Apex Harness Monitor[/bold cyan]  "
                  f"[dim]→ {server_url} | Ctrl-C para sair[/dim]\n")

    with Live(console=console, refresh_per_second=2, screen=False) as live:
        try:
            while True:
                data = _fetch(status_url, timeout=3.0)
                live.update(build_dashboard(data, server_url))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass

    console.print("\n[dim]Monitor encerrado.[/dim]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor ao vivo do Apex Harness Web Server."
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:7860",
        help="URL base do servidor (default: http://127.0.0.1:7860)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Porta (atalho para --url http://127.0.0.1:PORT)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Intervalo de atualização em segundos (default: 1.0)",
    )
    args = parser.parse_args()

    url = args.url
    if args.port:
        url = f"http://127.0.0.1:{args.port}"

    run(server_url=url, interval=args.interval)


if __name__ == "__main__":
    main()
