import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.request
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from apex_harness.core import ApexAgent
from apex_harness.tools import TOOLS_DEFINITION
from apex_harness.mcp_client import get_mcp_manager

console = Console()


class ThinkingStreamer:
    """
    Controla a animação em tempo real de pensamento (<think>...</think>) e
    a transição fluida para a resposta do assistente.
    """
    def __init__(self, console: Console, show_thinking: bool = False):
        self.console = console
        self.show_thinking = show_thinking
        self.start_time = time.time()
        self.think_start = None
        self.in_think = False
        self.had_think = False
        self.first_content_token = True
        self.buf = ""
        self.thought_chars = 0
        self.status = console.status("[bold cyan]Pensando...[/bold cyan]", spinner="dots")
        self.status.start()

    def pause(self):
        if self.status:
            try:
                self.status.stop()
            except Exception:
                pass

    def resume(self, msg: str = "[bold cyan]Pensando...[/bold cyan]"):
        if self.status and (self.in_think or self.first_content_token):
            try:
                self.status.update(msg)
                self.status.start()
            except Exception:
                pass

    def on_chunk(self, chunk: str):
        self.buf += chunk
        elapsed = time.time() - self.start_time

        # 1. Detecção do início de <think>
        if "<think>" in self.buf and not self.in_think:
            self.in_think = True
            self.had_think = True
            self.think_start = time.time()
            _, after = self.buf.split("<think>", 1)
            self.buf = after
            if self.status:
                self.status.update(f"[bold cyan]Pensando...[/bold cyan] [dim]({elapsed:.1f}s)[/dim]")
            if self.show_thinking:
                self.pause()
                self.console.print("\n[dim italic]🧠 Raciocínio:[/dim italic]")
            return

        # 2. Enquanto estiver no bloco <think>
        if self.in_think:
            if "</think>" in self.buf:
                thought_chunk, after = self.buf.split("</think>", 1)
                self.buf = after
                self.in_think = False
                self.pause()
                total_think = time.time() - (self.think_start or self.start_time)
                if self.show_thinking:
                    if thought_chunk:
                        self.console.print(f"[dim italic]{thought_chunk}[/dim italic]", end="")
                    self.console.print(f"\n[dim italic]└── 🧠 Pensou por {total_think:.1f}s[/dim italic]\n")
                else:
                    self.console.print(f"\n[dim italic]🧠 Pensou por {total_think:.1f}s[/dim italic]\n")
                
                self.console.print("[bold cyan]Apex:[/bold cyan] ", end="")
                if self.buf:
                    clean_after = self.buf.lstrip()
                    if clean_after:
                        self.console.out(clean_after, end="", highlight=False)
                    self.buf = ""
                return
            else:
                if self.show_thinking:
                    self.console.out(chunk, end="", highlight=False)
                else:
                    self.thought_chars += len(chunk)
                    if self.thought_chars % 25 == 0 and self.status:
                        self.status.update(f"[bold cyan]Pensando...[/bold cyan] [dim]({elapsed:.1f}s)[/dim]")
                return

        # 3. Resposta regular
        if self.first_content_token:
            self.first_content_token = False
            self.pause()
            if not self.had_think:
                self.console.print("\n[bold cyan]Apex:[/bold cyan] ", end="")

        if self.buf:
            self.console.out(self.buf, end="", highlight=False)
            self.buf = ""
        else:
            self.console.out(chunk, end="", highlight=False)

    def finish(self):
        self.pause()
        if self.first_content_token and self.buf:
            if not self.had_think:
                self.console.print("\n[bold cyan]Apex:[/bold cyan] ", end="")
            self.console.out(self.buf, end="", highlight=False)
            self.buf = ""

BANNER = """
[bold cyan]  █████╗ ██████╗ ███████╗██╗  ██╗[/bold cyan]  [bold white]APEX HARNESS[/bold white] [dim](Claude Code & MCP Edition)[/dim]
[bold cyan] ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝[/bold cyan]  [dim]Autonomous AI Engineer & Researcher[/dim]
[bold cyan] ███████║██████╔╝█████╗   ╚███╔╝ [/bold cyan]  [green]● AMD Ryzen AI 9 HX 370 | Radeon 890M[/green]
[bold cyan] ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗ [/bold cyan]  [yellow]● 96GB Unified Memory | Flash-Attention[/yellow]
[bold cyan] ██║  ██║██║     ███████╗██╔╝ ██╗[/bold cyan]  [magenta]● Live Web Access | Native MCP Tools Active[/magenta]
"""

def print_help():
    table = Table(title="Claude Code & Apex Harness Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green", width=22)
    table.add_column("Description", style="white")
    table.add_row("/help", "Exibe esta tela com todos os comandos disponíveis")
    table.add_row("/effort [low|med|high|off]", "Ajusta o nível de raciocínio (effort) do modelo")
    table.add_row("/think [on|off]", "Alterna exibição detalhada do raciocínio (<think>)")
    table.add_row("/clear, /new", "Limpa o histórico da sessão e inicia um novo tópico")
    table.add_row("/compact", "Compacta o contexto anterior para economizar tokens de memória")
    table.add_row("/doctor", "Verifica a saúde do sistema (Servidor, GPU, RAM, MCP, Ferramentas)")
    table.add_row("/tdp <task>", "Pipeline Context! (TDP): Researcher + Planner (PTCF) + Executor/Healer")
    table.add_row("/rag [ingest|search|ask] <args>", "Hybrid RAG + Rerank local sobre documentos e código")
    table.add_row("/bench [model]", "Mede TTFT, PP t/s e TG t/s com isolamento de warm-up")
    table.add_row("/mcp [load|unload]", "Lista, ativa (/mcp load) ou desativa (/mcp unload) ferramentas MCP")
    table.add_row("/init", "Cria o arquivo APEX.md com instruções e regras no projeto atual")
    table.add_row("/review", "Analisa e revisa alterações não commitadas (git diff) no projeto")
    table.add_row("/commit", "Gera uma mensagem de commit inteligente e commita via Git")
    table.add_row("/cost", "Exibe estimativas de tokens consumidos e tamanho de contexto")
    table.add_row("/tools", "Lista todas as ferramentas disponíveis (Built-in + MCP)")
    table.add_row("/status", "Exibe endpoint ativo, modelo carregado e pasta atual")
    table.add_row("/exit, /quit", "Encerra a sessão do Apex Harness")
    console.print(table)

def print_tools(agent: ApexAgent):
    table = Table(title="Ferramentas Disponíveis (Built-in & MCP)", show_header=True, header_style="bold magenta")
    table.add_column("Ferramenta", style="bold green", width=26)
    table.add_column("Tipo", style="yellow", width=10)
    table.add_column("Descrição", style="white")
    
    for t in agent.tools:
        fn = t["function"]
        name = fn["name"]
        tool_type = "MCP" if name.startswith("mcp_") else "Built-in"
        table.add_row(name, tool_type, fn.get("description", ""))
    console.print(table)

def print_mcp_servers(agent: ApexAgent):
    mgr = get_mcp_manager()
    table = Table(title="🔌 Servidores MCP Configurados", show_header=True, header_style="bold cyan")
    table.add_column("Servidor", style="bold green", width=22)
    table.add_column("Comando", style="white", width=36)
    table.add_column("Ferramentas Carregadas", style="yellow")
    
    if not mgr.servers:
        console.print(f"[dim]Nenhum servidor MCP configurado no arquivo ({mgr.config_path}).[/dim]")
        return

    for s_name, conf in mgr.servers.items():
        cmd_str = f"{conf.get('command', '')} {' '.join(conf.get('args', []))}"[:40]
        matching = [t for t in agent.tools if t["function"]["name"].startswith(f"mcp_{s_name}_")]
        status_str = f"[green]✓ {len(matching)} ativas[/green]" if matching else "[dim]Conectável sob demanda[/dim]"
        table.add_row(s_name, cmd_str, status_str)
    console.print(table)
    console.print(f"[dim]Configuração lida de: {mgr.config_path}[/dim]\n")

def run_doctor(agent: ApexAgent):
    table = Table(title="Apex Doctor — Diagnóstico do Sistema", show_header=True, header_style="bold yellow")
    table.add_column("Componente", style="bold white", width=24)
    table.add_column("Status", style="bold", width=14)
    table.add_column("Detalhe", style="dim white")

    # 1. Server Connection
    t0 = time.time()
    try:
        req = urllib.request.urlopen(f"{agent.base_url}/models", timeout=3)
        latency = (time.time() - t0) * 1000
        table.add_row("llama-server (Porta 8080)", "[green]ONLINE[/green]", f"Latência: {latency:.1f}ms | Endpoint: {agent.base_url}")
    except Exception as e:
        table.add_row("llama-server (Porta 8080)", "[yellow]OFFLINE / IDLE[/yellow]", f"Endpoint: {agent.base_url}")

    # 2. Hardware / GPU — reads DEVICE_LOCAL carve-out from sysfs
    vram_gib = 0.0
    try:
        import glob
        for p in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
            try:
                vram_gib = int(open(p).read().strip()) / 1024**3
                break
            except Exception:
                pass
    except Exception:
        pass
    try:
        gpu_info = subprocess.check_output(["lspci"], text=True)
        is_radeon = "Radeon" in gpu_info or "gfx" in gpu_info
        vram_str = f"{vram_gib:.0f} GiB DEVICE_LOCAL" if vram_gib > 0 else "—"
        table.add_row(
            "GPU Radeon 890M",
            "[green]ATIVO[/green]" if is_radeon else "[yellow]DETECTADO[/yellow]",
            f"RDNA 3.5 (16 CUs) | Vulkan | VRAM carve-out: {vram_str}"
        )
    except Exception:
        table.add_row("GPU Radeon 890M", "[yellow]INFO[/yellow]",
                      f"AMD Radeon 890M | VRAM carve-out: {vram_gib:.0f} GiB" if vram_gib else "AMD Radeon 890M")

    # 3. Memória RAM
    try:
        mem_info = subprocess.check_output(["free", "-h"], text=True).splitlines()[1].split()
        total_mem = mem_info[1]
        avail_mem = mem_info[6]
        table.add_row("Memória RAM", "[green]OK[/green]", f"Total: {total_mem} | Disponível: {avail_mem}")
    except Exception:
        table.add_row("Memória RAM", "[green]OK[/green]", "96GB RAM Unificada")

    # 4. MCP Servers
    try:
        mgr = get_mcp_manager()
        server_count = len(mgr.servers)
        table.add_row("Servidores MCP", f"[green]{server_count} CONFIGURADOS[/green]", f"{', '.join(mgr.servers.keys())}")
    except Exception as e:
        table.add_row("Servidores MCP", "[yellow]AVISO[/yellow]", str(e)[:50])

    # 5. Git Repository
    in_git = os.path.exists(".git")
    table.add_row("Repositório Git", "[green]SIM[/green]" if in_git else "[dim]NÃO[/dim]", str(Path.cwd()))

    # 6. Total Tools
    table.add_row("Total de Ferramentas", f"[green]{len(agent.tools)} CARREGADAS[/green]", "Built-in (web, bash, os) + MCP")

    console.print(table)

def run_init():
    target = Path.cwd() / "APEX.md"
    if target.exists():
        console.print(f"[yellow]O arquivo {target.name} já existe neste diretório.[/yellow]")
        return

    content = f"""# Instruções do Projeto (Apex Harness / Claude Code)

## 📌 Visão Geral
- **Projeto:** {Path.cwd().name}
- **Ambiente:** Linux (AMD Ryzen AI 9 HX 370 + Radeon 890M + 96GB RAM)

## 🛠️ Regras e Padrões de Código
1. **Raciocínio Cirúrgico:** Modifique apenas arquivos relevantes para a tarefa solicitada.
2. **Validação:** Sempre execute testes ou comandos de verificação após editar código.
3. **Estilo:** Código limpo, modular e bem documentado.
"""
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓ Arquivo [bold]APEX.md[/bold] inicializado com sucesso em {Path.cwd()}![/green]")
    except Exception as e:
        console.print(f"[red]Erro ao criar APEX.md: {e}[/red]")

def run_review(agent: ApexAgent):
    if not os.path.exists(".git"):
        console.print("[yellow]Este diretório não é um repositório Git (.git não encontrado).[/yellow]")
        return

    try:
        diff = subprocess.check_output(["git", "diff"], text=True).strip()
        if not diff:
            stat = subprocess.check_output(["git", "status", "-s"], text=True).strip()
            if not stat:
                console.print("[green]✓ Nenhuma alteração pendente no repositório (Working tree clean).[/green]")
                return
            diff = f"Arquivos modificados ou untracked:\n{stat}"

        console.print("[bold cyan]Revisando alterações locais com o Apex...[/bold cyan]\n")
        review_prompt = f"Analise o seguinte git diff e faça uma revisão de código técnica, apontando possíveis bugs, pontos de melhoria e elegância:\n\n```diff\n{diff[:4000]}\n```"
        
        console.print("[bold cyan]Apex:[/bold cyan] ", end="")
        agent.step(review_prompt, on_chunk=lambda c: console.print(c, end=""))
        console.print("\n")
    except Exception as e:
        console.print(f"[red]Erro ao executar git diff: {e}[/red]")

def run_commit(agent: ApexAgent):
    if not os.path.exists(".git"):
        console.print("[yellow]Este diretório não é um repositório Git.[/yellow]")
        return

    try:
        stat = subprocess.check_output(["git", "status", "-s"], text=True).strip()
        if not stat:
            console.print("[green]Nada para commitar (Working tree clean).[/green]")
            return

        diff_summary = subprocess.check_output(["git", "diff", "--stat"], text=True).strip()
        console.print(f"[dim]{stat}[/dim]\n")
        
        prompt = f"Gere uma mensagem de commit no formato convencional (ex: feat: ..., fix: ..., refactor: ...) concisa em 1 linha baseada nestas alterações:\n{stat}\n{diff_summary}"
        
        with console.status("Gerando mensagem de commit...", spinner="dots"):
            commit_msg = agent.step(prompt).strip().replace('"', '').replace("`", "")
            if "\n" in commit_msg:
                commit_msg = commit_msg.splitlines()[0]

        console.print(f"[bold cyan]Mensagem sugerida:[/bold cyan] [bold white]{commit_msg}[/bold white]")
        confirm = console.input("[yellow]Deseja commitar todas as alterações com esta mensagem? (s/N): [/yellow]").strip().lower()
        if confirm in ["s", "sim", "y", "yes"]:
            subprocess.run(["git", "add", "-A"])
            res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
            if res.returncode == 0:
                console.print(f"[green]✓ Commit realizado com sucesso![/green]\n[dim]{res.stdout.strip()}[/dim]")
            else:
                console.print(f"[red]Erro ao commitar:[/red] {res.stderr}")
        else:
            console.print("[dim]Commit cancelado.[/dim]")
    except Exception as e:
        console.print(f"[red]Erro no git commit: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Apex Harness - Claude Code & MCP Edition")
    parser.add_argument("--url", type=str, default=os.environ.get("APEX_API_BASE", "http://127.0.0.1:8080/v1"), help="API Base URL")
    parser.add_argument("--model", type=str, default=os.environ.get("APEX_MODEL", "llama-local-model"), help="Model alias")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP tools loading")
    parser.add_argument("--effort", type=str, default="medium", choices=["low", "medium", "high", "off"], help="Reasoning effort level")
    parser.add_argument("--show-thinking", action="store_true", help="Display full thinking trace in terminal")
    args = parser.parse_args()

    console.print(BANNER)
    console.print(f"[dim]Endpoint: [bold green]{args.url}[/bold green] | Modelo: [bold yellow]{args.model}[/bold yellow] | Projeto: [bold cyan]{Path.cwd()}[/bold cyan][/dim]")
    console.print("[dim]Digite sua mensagem ou use [bold]/help[/bold] para comandos. Ctrl+C cancela, Ctrl+D sai.\n[/dim]")

    show_thinking = args.show_thinking
    agent = ApexAgent(
        base_url=args.url,
        model_name=args.model,
        temperature=args.temp,
        enable_mcp=not args.no_mcp,
        reasoning_effort=args.effort
    )
    if args.no_mcp:
        console.print("[bold green]⚡ Modo Turbo Ativo[/bold green] [dim](~4.2 t/s, prompt leve). Use [bold]/mcp load[/bold] para ativar ferramentas extras quando precisar.\n[/dim]")
    else:
        console.print(f"[bold cyan]🔌 Modo Completo MCP Ativo[/bold cyan] [dim]({len(agent.tools)} ferramentas). Use [bold]/mcp unload[/bold] para acelerar a velocidade de resposta.\n[/dim]")

    session = PromptSession(history=InMemoryHistory())

    prompt_style = Style.from_dict({
        'prompt': '#00afff bold',
    })

    while True:
        try:
            user_input = session.prompt([('class:prompt', 'apex ❯ ')], style=prompt_style).strip()
            if not user_input:
                continue

            # Commands Handling
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ["/exit", "/quit", "/q"]:
                    console.print("[yellow]Encerrando Apex Harness. Até logo![/yellow]")
                    sys.exit(0)
                elif cmd in ["/help", "/h"]:
                    print_help()
                    continue
                elif cmd in ["/effort", "/reasoning"]:
                    parts = user_input.strip().split()
                    if len(parts) > 1:
                        res = agent.set_reasoning_effort(parts[1])
                        console.print(f"[green]✓ {res}[/green]")
                    else:
                        curr = agent.reasoning_effort.upper()
                        console.print(Panel(
                            f"Nível atual de raciocínio (effort): [bold green]{curr}[/bold green]\n\n"
                            "Opções disponíveis:\n"
                            "  • [bold cyan]/effort low[/bold cyan]    - ⚡ Baixo: Raciocínio rápido e conciso (economiza tokens)\n"
                            "  • [bold cyan]/effort medium[/bold cyan] - ⚖️ Médio: Equilíbrio padrão recomendado\n"
                            "  • [bold cyan]/effort high[/bold cyan]   - 🧠 Alto: Raciocínio analítico profundo e minucioso\n"
                            "  • [bold cyan]/effort off[/bold cyan]    - 🚀 Desligado: Resposta direta sem bloco <think>",
                            title="🧠 Nível de Raciocínio (Reasoning Effort)",
                            border_style="cyan"
                        ))
                    continue
                elif cmd in ["/think", "/thought"]:
                    parts = user_input.strip().split()
                    if len(parts) > 1 and parts[1].lower() in ["on", "sim", "true", "1"]:
                        show_thinking = True
                        console.print("[green]✓ Exibição de raciocínio ativada. O texto de <think> será exibido em itálico.[/green]")
                    elif len(parts) > 1 and parts[1].lower() in ["off", "nao", "não", "false", "0"]:
                        show_thinking = False
                        console.print("[yellow]✓ Exibição de raciocínio ocultada (apenas animação '🧠 Pensou por X.Xs').[/yellow]")
                    else:
                        st = "ATIVADA" if show_thinking else "OCULTADA (apenas animação)"
                        console.print(f"[dim]Exibição do texto de pensamento: [bold]{st}[/bold]. Use [bold]/think on[/bold] ou [bold]/think off[/bold].[/dim]")
                    continue
                elif cmd in ["/mcp"]:
                    parts = user_input.strip().split()
                    if len(parts) > 1 and parts[1].lower() in ["load", "on", "enable", "carregar"]:
                        with console.status("[cyan]Carregando ferramentas MCP...[/cyan]"):
                            agent._init_mcp()
                        console.print(f"[green]✓ Ferramentas MCP carregadas! Total de ferramentas ativas: {len(agent.tools)}[/green]")
                    elif len(parts) > 1 and parts[1].lower() in ["unload", "off", "disable", "descarregar", "remover"]:
                        agent.tools = [t for t in agent.tools if not t["function"]["name"].startswith("mcp_")]
                        console.print(f"[yellow]✓ Ferramentas MCP descarregadas (Modo Turbo: ~4.2 t/s). Total de ferramentas: {len(agent.tools)}[/yellow]")
                    else:
                        print_mcp_servers(agent)
                        console.print("[dim]Dica: use [bold]/mcp load[/bold] para ativar ou [bold]/mcp unload[/bold] para acelerar.[/dim]")
                    continue
                elif cmd in ["/tools", "/t"]:
                    print_tools(agent)
                    continue
                elif cmd in ["/clear", "/new", "/reset"]:
                    agent.reset()
                    console.print("[green]✓ Contexto limpo. Nova conversa iniciada.[/green]")
                    continue
                elif cmd in ["/compact", "/c"]:
                    msg = agent.compact()
                    console.print(f"[green]✓ {msg}[/green]")
                    continue
                elif cmd in ["/doctor", "/diag"]:
                    run_doctor(agent)
                    continue
                elif cmd in ["/bench", "/benchmark"]:
                    # Auto-detect model path from the running server if not provided.
                    parts = user_input.strip().split(maxsplit=1)
                    model_arg = parts[1].strip() if len(parts) > 1 else None
                    if model_arg is None:
                        try:
                            import json as _json
                            import urllib.request as _req
                            raw = _req.urlopen(f"{agent.base_url}/models", timeout=3).read()
                            models_data = _json.loads(raw)
                            # Try to get the model id from the server's /v1/models response.
                            models_list = models_data.get("data", [])
                            model_arg = models_list[0].get("id", "") if models_list else ""
                        except Exception:
                            model_arg = ""
                    if not model_arg:
                        console.print(
                            "[yellow]Usage: /bench <path_to_gguf> [--quants Q4_K_M Q6_K] [--runs N][/yellow]\n"
                            "[dim]If the server is online, the model path is auto-detected.[/dim]"
                        )
                        continue
                    # Build the apex-bench command. Pass remaining args through.
                    bench_extra = parts[1].replace(model_arg, "", 1).strip() if len(parts) > 1 else ""
                    bench_cmd = [
                        sys.executable, "-m", "apex_harness.bench",
                        model_arg,
                    ]
                    if bench_extra:
                        import shlex
                        bench_cmd += shlex.split(bench_extra)
                    console.print(f"[bold cyan]Apex Bench[/bold cyan] [dim]{' '.join(bench_cmd[2:])}[/dim]")
                    try:
                        subprocess.run(bench_cmd, check=False)
                    except Exception as bench_err:
                        console.print(f"[red]Erro no bench: {bench_err}[/red]")
                    continue
                elif cmd in ["/tdp", "/context"]:
                    parts = user_input.strip().split(maxsplit=1)
                    task_prompt = parts[1].strip() if len(parts) > 1 else ""
                    if not task_prompt:
                        console.print(Panel(
                            "Uso: [bold cyan]/tdp <descrição da tarefa complexa>[/bold cyan]\n\n"
                            "O pipeline TDP (Context! / Task-Decoupled Planning) desacopla cognição em 3 agentes:\n"
                            "  1. 🔬 [bold cyan]Researcher[/bold cyan]: Destila restrições e contratos, eliminando até 90% do ruído.\n"
                            "  2. 📐 [bold magenta]Planner[/bold magenta]: Elabora blueprint de engenharia determinístico (Google PTCF).\n"
                            "  3. ⚡ [bold green]Executor & Healer[/bold green]: Gera código e valida em sandbox com auto-cura em caso de erro.\n\n"
                            "Exemplo: [italic]/tdp Crie um benchmark concorrente em Python com asyncio e rate-limiting[/italic]",
                            title="🚀 Context! TDP — Task-Decoupled Planning",
                            border_style="cyan"
                        ))
                        continue

                    console.print(Panel(
                        f"[bold white]{task_prompt}[/bold white]",
                        title="🚀 Context! TDP Pipeline Iniciado",
                        border_style="magenta"
                    ))

                    try:
                        from apex_harness.tdp import run_tdp, TDPConfig

                        cfg = TDPConfig(execute_code=True, stream=True)

                        def tdp_event_handler(stage: str, chunk: str):
                            if stage == "status":
                                console.print(chunk, end="")
                            elif stage == "researcher":
                                console.out(chunk, end="", style="dim cyan")
                            elif stage == "planner":
                                console.out(chunk, end="", style="white")
                            elif stage in ("executor", "healer"):
                                console.out(chunk, end="", style="green")

                        result = run_tdp(task=task_prompt, agent=agent, cfg=cfg, on_event=tdp_event_handler)

                        stat_color = "green" if result.healed or result.heal_attempts == 0 else "yellow"
                        console.print("\n")
                        console.print(Panel(
                            f"✔ [bold]Status:[/bold] Concluído com sucesso\n"
                            f"📉 [bold]Economia de Contexto:[/bold] ~{result.tokens_saved_pct:.0%}\n"
                            f"🔄 [bold]Auto-Cura (Healer):[/bold] {'Sim (' + str(result.heal_attempts) + ' correções)' if result.healed else ('0 erros (Direto na primeira)' if result.heal_attempts == 0 else 'Falhou')}",
                            title="🎉 TDP Execução Concluída",
                            border_style=stat_color
                        ))

                        agent.history.append({"role": "user", "content": f"[TDP Task]: {task_prompt}"})
                        agent.history.append({
                            "role": "assistant",
                            "content": f"### TDP Specification\n{result.spec}\n\n### Architectural Plan (PTCF)\n{result.plan}\n\n### Validated Solution\n{result.solution}"
                        })
                    except Exception as tdp_err:
                        console.print(f"[red]Erro no TDP pipeline:[/red] {tdp_err}")
                    continue
                elif cmd in ["/rag"]:
                    parts = user_input.strip().split(maxsplit=2)
                    subcmd = parts[1].lower() if len(parts) > 1 else "help"
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    if subcmd in ["help", "-h", "--help"]:
                        console.print(Panel(
                            "Comandos do [bold cyan]Apex RAG (Hybrid + Rerank)[/bold cyan]:\n\n"
                            "  • [bold cyan]/rag ingest <arquivo/pasta>[/bold cyan]  - Indexa documentos locais (Markdown, Código, Texto)\n"
                            "  • [bold cyan]/rag search <query>[/bold cyan]          - Busca híbrida (BM25 + Nomic Embeddings) com Rerank\n"
                            "  • [bold cyan]/rag ask <pergunta>[/bold cyan]           - Faz pergunta direta com contexto injetado no LLM",
                            title="📚 Apex RAG Engine",
                            border_style="cyan"
                        ))
                        continue

                    try:
                        from rag.engine import ApexRAG
                        rag_engine = ApexRAG()

                        if subcmd in ["ingest", "add"]:
                            target_path = arg or parts[1] if len(parts) > 1 and subcmd not in ["ingest", "add"] else arg
                            if not target_path:
                                console.print("[yellow]Uso: /rag ingest <caminho_arquivo_ou_pasta>[/yellow]")
                                continue
                            
                            p = Path(target_path).resolve()
                            with console.status(f"[cyan]Indexando {p.name} com nomic-embed-text...[/cyan]"):
                                if p.is_file():
                                    count = rag_engine.ingest_file(str(p))
                                    console.print(f"[green]✓ Arquivo '{p.name}' indexado com sucesso! ({count} chunks criados)[/green]")
                                elif p.is_dir():
                                    res = rag_engine.ingest_directory(str(p))
                                    total_chunks = sum(v for v in res.values() if isinstance(v, int))
                                    console.print(f"[green]✓ Pasta '{p.name}' indexada! ({len(res)} arquivos, {total_chunks} chunks)[/green]")
                                else:
                                    console.print(f"[red]Caminho não encontrado: {target_path}[/red]")
                            continue

                        elif subcmd in ["search", "find"]:
                            query = arg if arg else (parts[1] if len(parts) > 1 and subcmd not in ["search", "find"] else "")
                            if not query:
                                console.print("[yellow]Uso: /rag search <termo de busca>[/yellow]")
                                continue

                            with console.status("[cyan]Pesquisando via Hybrid RAG + Rerank...[/cyan]"):
                                results = rag_engine.search(query=query, mode="rerank", top_k=5)

                            if not results:
                                console.print(f"[yellow]Nenhum resultado encontrado para '{query}'.[/yellow]")
                            else:
                                console.print(f"\n[bold green]Top Resultados para:[/bold green] [italic]'{query}'[/italic]\n")
                                for idx, r in enumerate(results, 1):
                                    source = r.get("metadata", {}).get("source", r.get("doc_id", "Doc"))
                                    score = r.get("rerank_score", r.get("score", 0.0))
                                    console.print(Panel(
                                        r.get("content", "").strip(),
                                        title=f"[{idx}] {source} (Score: {score:.4f})",
                                        border_style="cyan"
                                    ))
                            continue

                        elif subcmd in ["ask", "q"]:
                            question = arg if arg else (parts[1] if len(parts) > 1 and subcmd not in ["ask", "q"] else "")
                            if not question:
                                console.print("[yellow]Uso: /rag ask <sua pergunta>[/yellow]")
                                continue

                            with console.status("[cyan]Recuperando contexto via RAG...[/cyan]"):
                                results = rag_engine.search(query=question, mode="rerank", top_k=4)

                            if not results:
                                console.print("[yellow]Nenhum documento relevante encontrado. Encaminhando sem RAG...[/yellow]")
                                user_input = question
                            else:
                                context_blocks = []
                                for idx, r in enumerate(results, 1):
                                    src = r.get("metadata", {}).get("source", r.get("doc_id", "doc"))
                                    context_blocks.append(f"--- Documento [{idx}] ({src}) ---\n{r.get('content', '').strip()}")
                                
                                context_str = "\n\n".join(context_blocks)
                                augmented_prompt = (
                                    f"Contexto relevante recuperado do Apex RAG:\n\n{context_str}\n\n"
                                    f"Com base nas informações acima e em seu conhecimento, responda à pergunta:\n{question}"
                                )
                                user_input = augmented_prompt
                                console.print(f"[dim green]✓ {len(results)} chunks de contexto injetados via Apex RAG.[/dim green]")
                                # Permitir que caia no fluxo padrão de geração do agente abaixo!
                                # Não chamamos continue aqui.

                    except Exception as rag_err:
                        console.print(f"[red]Erro no Apex RAG:[/red] {rag_err}")
                        continue

                elif cmd in ["/init"]:
                    run_init()
                    continue
                elif cmd in ["/review"]:
                    run_review(agent)
                    continue
                elif cmd in ["/commit"]:
                    run_commit(agent)
                    continue
                elif cmd in ["/cost", "/usage"]:
                    console.print(Panel(
                        f"Mensagens no histórico: {len(agent.history)}\n"
                        f"Ferramentas ativas: {len(agent.tools)}\n"
                        f"Tokens de saída estimados: ~{sum(len(m.get('content') or '') for m in agent.history) // 4}\n"
                        f"Custo Local: [bold green]R$ 0,00 (100% Gratuito na GPU Radeon 890M)[/bold green]",
                        title="Uso de Recursos & Contexto",
                        border_style="cyan"
                    ))
                    continue
                elif cmd in ["/status", "/s"]:
                    console.print(Panel(
                        f"Base URL: {agent.base_url}\nModelo: {agent.model_name}\nMensagens em memória: {len(agent.history)}\nFerramentas carregadas: {len(agent.tools)}\nProjeto: {Path.cwd()}",
                        title="Status do Apex",
                        border_style="cyan"
                    ))
                    continue
                else:
                    console.print(f"[red]Comando desconhecido: {user_input}. Digite /help para ver os comandos.[/red]")
                    continue

            streamer = ThinkingStreamer(console, show_thinking=show_thinking)

            # Tool Execution Callbacks
            def on_tool_start(name: str, tool_args: dict):
                streamer.pause()
                args_summary = ", ".join(f"{k}='{str(v)[:40]}...'" if len(str(v)) > 40 else f"{k}={v}" for k, v in tool_args.items())
                is_mcp = name.startswith("mcp_")
                icon = "🔌 MCP:" if is_mcp else "⚡ Ferramenta:"
                console.print(f"  [bold yellow]{icon}[/bold yellow] [bold cyan]{name}[/bold cyan]({args_summary})")

            def on_tool_finish(name: str, result: str):
                lines = result.strip().splitlines()
                preview = lines[0][:100] + ("..." if len(lines) > 1 or len(lines[0]) > 100 else "")
                console.print(f"  [dim green]✔ Retorno ({name}):[/dim green] [dim]{preview}[/dim]")
                streamer.resume(f"[bold cyan]Processando retorno de {name}...[/bold cyan]")

            # Real-time streaming output with thinking animation
            try:
                response = agent.step(
                    user_input=user_input,
                    on_tool_start=on_tool_start,
                    on_tool_finish=on_tool_finish,
                    on_chunk=streamer.on_chunk
                )
            finally:
                streamer.finish()
            console.print("\n")

        except KeyboardInterrupt:
            console.print("\n[dim yellow]Operação cancelada pelo utilizador.[/dim yellow]\n")
            continue
        except EOFError:
            console.print("\n[yellow]Até logo![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Erro: {str(e)}[/red]\n")

if __name__ == "__main__":
    main()
