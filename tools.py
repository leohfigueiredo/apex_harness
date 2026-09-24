import os
import subprocess
import urllib.request
import urllib.parse
import re
import html
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo to get fresh information, documentation, news, or technical solutions."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers=headers)
        html_content = urllib.request.urlopen(req, timeout=12).read().decode('utf-8', errors='ignore')

        titles = re.findall(r'<h2 class=\"result__title\">.*?<a[^>]*>(.*?)</a>', html_content, re.DOTALL)
        snippets = re.findall(r'<a class=\"result__snippet[^\"]*\"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        links = re.findall(r'<a class=\"result__url\" href=\"([^\"]+)\"', html_content)

        results = []
        for i in range(min(len(links), len(snippets), max_results * 2)):
            raw_url = links[i]
            if 'ad_domain' in raw_url or 'bing.com' in raw_url:
                continue
            if 'uddg=' in raw_url:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                target_url = qs.get('uddg', [raw_url])[0]
            else:
                target_url = raw_url
            if target_url.startswith('//'):
                target_url = 'https:' + target_url

            t = html.unescape(re.sub(r'<.*?>', '', titles[i])).strip() if i < len(titles) else 'No Title'
            s = html.unescape(re.sub(r'<.*?>', '', snippets[i])).strip()
            
            results.append(f"- **Title**: {t}\n  **URL**: {target_url}\n  **Snippet**: {s}")
            if len(results) >= max_results:
                break

        if not results:
            # Fallback to instant answer API
            api_url = 'https://api.duckduckgo.com/?q=' + urllib.parse.quote(query) + '&format=json'
            api_req = urllib.request.Request(api_url, headers={'User-Agent': 'ApexHarness/2.0'})
            data = json.loads(urllib.request.urlopen(api_req, timeout=8).read().decode('utf-8', errors='ignore'))
            abstract = data.get('AbstractText')
            if abstract:
                return f"**Direct Answer**:\n{abstract}\n\n**Source**: {data.get('AbstractURL', 'N/A')}"
            return f"No results found for query: '{query}'"

        return "\n\n".join(results)
    except Exception as e:
        return f"Error during web search: {str(e)}"

def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch the contents of a webpage and extract clean readable text."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'
        })
        raw = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
        
        # Remove noisy tags
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'svg', 'noscript']:
            raw = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            
        raw = re.sub(r'<(br|p|div|h[1-6]|li|tr)[^>]*>', '\n', raw, flags=re.IGNORECASE)
        text = re.sub(r'<.*?>', '', raw)
        text = html.unescape(text)
        
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = '\n'.join(lines)
        if len(cleaned) > max_chars:
            return cleaned[:max_chars] + f"\n\n... [Truncated: showing first {max_chars} characters]"
        return cleaned if cleaned else "Webpage returned empty text content."
    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"

def list_dir(path: str = ".", max_depth: int = 2) -> str:
    """List contents of a directory with file sizes and directory indicators."""
    try:
        base = Path(path).resolve()
        if not base.exists():
            return f"Error: Path '{path}' does not exist."
        if not base.is_dir():
            return f"Error: Path '{path}' is a file, not a directory."

        entries = []
        def scan(current: Path, depth: int):
            if depth > max_depth:
                return
            try:
                items = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                entries.append(f"{'  ' * depth}[Permission Denied] {current.name}/")
                return

            for item in items:
                rel = item.relative_to(base)
                indent = "  " * (len(rel.parts) - 1)
                if item.name.startswith('.') and item.name not in ['.env', '.gitignore']:
                    continue
                if item.is_dir():
                    entries.append(f"{indent}📁 {item.name}/")
                    scan(item, depth + 1)
                else:
                    try:
                        sz = item.stat().st_size
                        if sz < 1024:
                            sz_str = f"{sz} B"
                        elif sz < 1024 * 1024:
                            sz_str = f"{sz/1024:.1f} KB"
                        else:
                            sz_str = f"{sz/(1024*1024):.1f} MB"
                        entries.append(f"{indent}📄 {item.name} ({sz_str})")
                    except Exception:
                        entries.append(f"{indent}📄 {item.name}")

        scan(base, 1)
        return f"Directory listing of: {base}\n" + ("\n".join(entries) if entries else "(Empty directory)")
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"

def read_file(path: str, start_line: int = 1, end_line: int = -1, max_window: int = 250, max_chars: int = 16000) -> str:
    """Read contents of a text file or PDF with line numbers and safety limits to prevent context blowout."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        if not p.is_file():
            return f"Error: '{path}' is not a regular file."

        # Native support for PDF files via pdftotext
        if p.suffix.lower() == ".pdf":
            try:
                res = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, timeout=10)
                if res.returncode == 0 and res.stdout.strip():
                    lines = res.stdout.splitlines(keepends=True)
                else:
                    return f"Error: Could not extract text from PDF '{path}'."
            except Exception as pe:
                return f"Error reading PDF '{path}': {str(pe)}"
        else:
            # Check for binary file (null bytes or binary extensions)
            bin_exts = {".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz", ".gguf", ".bin", ".so", ".exe", ".iso"}
            if p.suffix.lower() in bin_exts:
                return f"Error: '{path}' is a binary file. Text inspection is disabled to prevent context corruption."
            try:
                with open(p, 'rb') as fb:
                    if b'\x00' in fb.read(2048):
                        return f"Error: '{path}' contains binary null bytes. Reading as text is disabled."
            except Exception:
                pass

            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        total = len(lines)
        s = max(1, start_line)
        
        # Automatic pagination if end_line is unspecified and file is large
        if end_line <= 0:
            e = min(total, s + max_window - 1)
        else:
            e = min(total, end_line)

        numbered = [f"{i:4d} | {lines[i-1]}" for i in range(s, e + 1)]
        
        notice = ""
        if e < total:
            notice = f"\n... [Remaining {total - e} lines omitted. Use start_line={e+1} to view more]"
            
        header = f"File: {p} (Lines {s}-{e} of {total})\n"
        out_text = header + "".join(numbered) + notice
        if len(out_text) > max_chars:
            out_text = out_text[:max_chars] + f"\n\n... [Truncated to {max_chars} chars to prevent context blowout]"
        return out_text
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with new content. Automatically creates parent directories."""
    try:
        p = Path(path).resolve()
        from apex_harness.sandbox import get_sandbox_executor
        sandbox = get_sandbox_executor()

        # Check sandbox confinement if APEX_SANDBOX is enabled
        if os.environ.get("APEX_SANDBOX", "0").lower() in ("1", "true", "yes"):
            if not sandbox.is_path_safe(str(p)):
                return f"Security Error: Attempted write outside designated workspace ({p}). Blocked by Sandbox."

        diff = sandbox.generate_diff(str(p), content)

        # Dry-run check
        if sandbox.config.dry_run:
            return f"[DRY-RUN] File write simulated for {p}.\n\nDiff Preview:\n{diff if diff else '(No changes)'}"

        # Self-Critic check if APEX_CRITIC=1
        if os.environ.get("APEX_CRITIC", "0").lower() in ("1", "true", "yes"):
            from apex_harness.critic import run_critic
            crit_res = run_critic(diff)
            if not crit_res.approved:
                return f"[CRITIC REJECTED] Review identified potential issues:\n{crit_res.feedback}\n\nOperation aborted."

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)

        # Log touched file in session memory
        try:
            from apex_harness.session_memory import get_session_memory
            sess_id = os.environ.get("APEX_SESSION_ID", "default_session")
            get_session_memory().log_file_touched(sess_id, str(p))
        except Exception:
            pass

        return f"Successfully written {len(content)} characters to {p}"
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"

def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace a specific block of text in a file with new text, with newline & whitespace tolerance."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        new_content = None

        # 1. Direct match
        if old_text in content:
            occurrences = content.count(old_text)
            if occurrences > 1:
                return f"Warning: 'old_text' matched {occurrences} times. Please provide more surrounding lines for unique matching."
            new_content = content.replace(old_text, new_text, 1)

        # 2. Line-ending normalized match (\r\n vs \n)
        if new_content is None:
            norm_content = content.replace('\r\n', '\n')
            norm_old = old_text.replace('\r\n', '\n')
            norm_new = new_text.replace('\r\n', '\n')

            if norm_old in norm_content:
                occurrences = norm_content.count(norm_old)
                if occurrences > 1:
                    return f"Warning: 'old_text' matched {occurrences} times after newline normalization. Provide more context."
                new_content = norm_content.replace(norm_old, norm_new, 1)

        # 3. Trailing-whitespace tolerant matching
        if new_content is None:
            norm_content = content.replace('\r\n', '\n')
            norm_old = old_text.replace('\r\n', '\n')
            norm_new = new_text.replace('\r\n', '\n')
            content_lines = norm_content.splitlines()
            old_lines = [l.rstrip() for l in norm_old.splitlines() if l.strip()]
            
            if old_lines:
                matched_start = -1
                match_len = len(old_lines)
                for idx in range(len(content_lines) - match_len + 1):
                    window = [content_lines[idx + j].rstrip() for j in range(match_len)]
                    if window == old_lines:
                        if matched_start != -1:
                            return f"Warning: multiple fuzzy matches found in {p}. Please provide more unique context lines."
                        matched_start = idx

                if matched_start != -1:
                    replacement_lines = norm_new.splitlines()
                    updated_lines = content_lines[:matched_start] + replacement_lines + content_lines[matched_start + match_len:]
                    new_content = "\n".join(updated_lines) + ("\n" if content.endswith("\n") else "")

        if new_content is None:
            return f"Error: 'old_text' not found in {p}. Verify indentation, special characters and exact lines."

        from apex_harness.sandbox import get_sandbox_executor
        sandbox = get_sandbox_executor()

        # Confinement check if APEX_SANDBOX is enabled
        if os.environ.get("APEX_SANDBOX", "0").lower() in ("1", "true", "yes"):
            if not sandbox.is_path_safe(str(p)):
                return f"Security Error: Attempted edit outside designated workspace ({p}). Blocked by Sandbox."

        diff = sandbox.generate_diff(str(p), new_content)

        # Dry run check
        if sandbox.config.dry_run:
            return f"[DRY-RUN] File edit simulated for {p}.\n\nDiff Preview:\n{diff}"

        # Self-Critic check if APEX_CRITIC=1
        if os.environ.get("APEX_CRITIC", "0").lower() in ("1", "true", "yes"):
            from apex_harness.critic import run_critic
            crit_res = run_critic(diff)
            if not crit_res.approved:
                return f"[CRITIC REJECTED] Review identified potential issues:\n{crit_res.feedback}\n\nOperation aborted."

        with open(p, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Log touched file in session memory
        try:
            from apex_harness.session_memory import get_session_memory
            sess_id = os.environ.get("APEX_SESSION_ID", "default_session")
            get_session_memory().log_file_touched(sess_id, str(p))
        except Exception:
            pass

        return f"Successfully updated {p}"
    except Exception as e:
        return f"Error editing file '{path}': {str(e)}"

def bash_exec(command: str, timeout: int = 60) -> str:
    """Execute a bash command in the terminal and return stdout + stderr."""
    use_sandbox = os.environ.get("APEX_SANDBOX", "0").lower() in ("1", "true", "yes")
    dry_run = os.environ.get("APEX_DRY_RUN", "0").lower() in ("1", "true", "yes")
    if use_sandbox or dry_run:
        from apex_harness.sandbox import get_sandbox_executor
        executor = get_sandbox_executor()
        res = executor.execute(command, timeout=timeout)
        return res.to_tool_output()

    try:
        res = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = []
        if res.stdout:
            out.append(f"STDOUT:\n{res.stdout.strip()}")
        if res.stderr:
            out.append(f"STDERR:\n{res.stderr.strip()}")
        out.append(f"Exit Code: {res.returncode}")
        return "\n\n".join(out)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing bash command: {str(e)}"

_CURRENT_AGENT: Any = None

def set_current_agent(agent: Any) -> None:
    """Store the current active ApexAgent instance for tools that need agent cognition."""
    global _CURRENT_AGENT
    _CURRENT_AGENT = agent

def get_current_agent() -> Any:
    """Retrieve the current active ApexAgent instance."""
    return _CURRENT_AGENT

def run_tdp_pipeline(task: str, execute_code: bool = True) -> str:
    """
    Run the Task-Decoupled Planning (TDP) pipeline for complex tasks.
    Decomposes the task into Researcher (spec distillation), Planner (PTCF blueprint),
    and Executor (code + self-healing validation).
    """
    try:
        from apex_harness.tdp import run_tdp, TDPConfig
        agent = get_current_agent()
        if agent is None:
            from apex_harness.core import ApexAgent
            agent = ApexAgent()

        cfg = TDPConfig(execute_code=execute_code, stream=False)
        result = run_tdp(task=task, agent=agent, cfg=cfg)

        output = [
            "=== TDP SPECIFICATION (RESEARCHER) ===",
            result.spec.strip(),
            "",
            "=== TDP ARCHITECTURAL PLAN (PLANNER) ===",
            result.plan.strip(),
            "",
            "=== TDP SOLUTION (EXECUTOR) ===",
            result.solution.strip(),
            "",
            f"[TDP Stats: Self-Healed={result.healed} ({result.heal_attempts} retries), Estimated Context Saved={result.tokens_saved_pct:.0%}]"
        ]
        return "\n".join(output)
    except Exception as e:
        return f"Error executing TDP pipeline: {str(e)}"


def rag_search(query: str, mode: str = "hybrid", top_k: int = 5) -> str:
    """Search locally indexed documents, code, and knowledge using Hybrid RAG + Rerank."""
    try:
        from rag.engine import ApexRAG
        engine = ApexRAG()
        results = engine.search(query=query, mode=mode, top_k=top_k)
        if not results:
            return f"No relevant indexed documents found for query: '{query}'"
        
        output = [f"### Apex RAG Search Results (mode: {mode})"]
        for idx, item in enumerate(results, start=1):
            source = item.get("metadata", {}).get("source", item.get("doc_id", "Unknown"))
            score = item.get("rerank_score", item.get("rrf_score", item.get("score", 0.0)))
            output.append(f"**[{idx}] Source:** `{source}` (Score: {score:.4f})")
            output.append(f"```text\n{item.get('content', '').strip()}\n```\n")
        return "\n".join(output)
    except Exception as e:
        return f"Error during rag_search: {str(e)}"


def rag_ingest(path: str) -> str:
    """Ingest a file or directory into the Apex RAG knowledge base."""
    try:
        from rag.engine import ApexRAG
        engine = ApexRAG()
        p = Path(path).resolve()
        if p.is_file():
            count = engine.ingest_file(str(p))
            return f"Successfully ingested file '{p.name}' ({count} chunks created and embedded)."
        elif p.is_dir():
            results = engine.ingest_directory(str(p))
            total_chunks = sum(v for v in results.values() if isinstance(v, int))
            return f"Successfully ingested directory '{p.name}' ({len(results)} files, {total_chunks} chunks total)."
        else:
            return f"Path not found: {path}"
    except Exception as e:
        return f"Error during rag_ingest: {str(e)}"


def recall_memory(query: str, top_k: int = 5) -> str:
    """Recall past conversation context, notes, or archived session history using semantic RAG search."""
    try:
        from apex_harness.rag.store import VectorStore
        from apex_harness.rag.embeddings import OllamaEmbedder
        
        db_path = os.environ.get("APEX_RAG_DB", "apex_rag.db")
        if not os.path.exists(db_path):
            return "No archived session memory database found."
            
        store = VectorStore(db_path)
        embedder = OllamaEmbedder()
        q_embs = embedder.get_embeddings_batch([query])
        if not q_embs or not q_embs[0]:
            return f"No embeddings generated for query: '{query}'"
            
        results = store.dense_search(q_embs[0], top_k=top_k)
        if not results:
            return f"No relevant past memory found for query: '{query}'"
            
        snippets = []
        for r in results:
            meta = r.get("metadata", {})
            doc_id = meta.get("doc_id", "session_memory")
            snippets.append(f"[{doc_id} (Score: {r['score']:.2f})]\n{r['content']}")
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error recalling memory: {str(e)}"


def git_status(repo_path: str = ".") -> str:
    """Show the working tree status of a git repository."""
    try:
        r_path = Path(repo_path).resolve()
        res = subprocess.run(
            ["git", "-C", str(r_path), "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if res.returncode != 0:
            return f"Git error (code {res.returncode}): {res.stderr.strip()}"
        out = res.stdout.strip()
        return out if out else "Working tree clean (no changes)."
    except Exception as e:
        return f"Error executing git status: {str(e)}"


def git_diff(repo_path: str = ".", staged: bool = False, file_path: str = "") -> str:
    """Show changes between commits, commit and working tree, etc."""
    try:
        r_path = Path(repo_path).resolve()
        cmd = ["git", "-C", str(r_path), "diff"]
        if staged:
            cmd.append("--staged")
        if file_path:
            cmd.extend(["--", file_path])

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if res.returncode != 0:
            return f"Git diff error (code {res.returncode}): {res.stderr.strip()}"
        out = res.stdout
        if not out:
            return "No diff found (working tree matches index/target)."
        if len(out) > 16000:
            return out[:16000] + "\n\n... [Truncated: diff exceeded 16000 chars]"
        return out
    except Exception as e:
        return f"Error executing git diff: {str(e)}"


def git_log(repo_path: str = ".", n: int = 10) -> str:
    """Show recent commit logs in a concise oneline format."""
    try:
        r_path = Path(repo_path).resolve()
        res = subprocess.run(
            ["git", "-C", str(r_path), "log", f"-n{n}", "--oneline", "--decorate"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if res.returncode != 0:
            return f"Git log error (code {res.returncode}): {res.stderr.strip()}"
        out = res.stdout.strip()
        return out if out else "No commits found in repository."
    except Exception as e:
        return f"Error executing git log: {str(e)}"


def git_branch(repo_path: str = ".", create: str = "", switch_to: str = "") -> str:
    """List, create, or switch git branches."""
    try:
        r_path = Path(repo_path).resolve()
        if create:
            res = subprocess.run(["git", "-C", str(r_path), "checkout", "-b", create], capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return f"Git branch error: {res.stderr.strip()}"
            return f"Created and switched to branch '{create}'."
        elif switch_to:
            res = subprocess.run(["git", "-C", str(r_path), "checkout", switch_to], capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return f"Git checkout error: {res.stderr.strip()}"
            return f"Switched to branch '{switch_to}'."
        else:
            res = subprocess.run(["git", "-C", str(r_path), "branch", "-a"], capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return f"Git branch error: {res.stderr.strip()}"
            return res.stdout.strip()
    except Exception as e:
        return f"Error executing git branch: {str(e)}"


def git_commit(repo_path: str = ".", message: str = "", add_all: bool = False) -> str:
    """Record changes to the repository."""
    try:
        if not message:
            return "Error: Commit message cannot be empty."
        r_path = Path(repo_path).resolve()
        if add_all:
            add_res = subprocess.run(["git", "-C", str(r_path), "add", "-A"], capture_output=True, text=True, timeout=15)
            if add_res.returncode != 0:
                return f"Git add error: {add_res.stderr.strip()}"

        res = subprocess.run(["git", "-C", str(r_path), "commit", "-m", message], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return f"Git commit error (code {res.returncode}): {res.stderr.strip()}"
        return res.stdout.strip()
    except Exception as e:
        return f"Error executing git commit: {str(e)}"


def session_log(event_type: str, content: str) -> str:
    """Log an architectural decision, touched file, open TODO, or note into persistent session memory."""
    try:
        from apex_harness.session_memory import get_session_memory
        session_id = os.environ.get("APEX_SESSION_ID", "default_session")
        mem = get_session_memory()
        ev_id = mem.log_event(session_id=session_id, event_type=event_type, content=content)
        return f"Successfully logged session event #{ev_id} ({event_type}): {content[:100]}"
    except Exception as e:
        return f"Error logging session event: {str(e)}"


def consult_wiki(query: str, category: str = "") -> str:
    """Search and read knowledge articles from the persistent WikiSkill repository."""
    try:
        from apex_harness.wikiskill import consult_wiki as _cw
        return _cw(query=query, category=category if category else None)
    except Exception as e:
        return f"Error consulting wiki: {str(e)}"


def record_wiki_skill(title: str, content: str, category: str = "workflows", tags: Optional[List[str]] = None) -> str:
    """Save an architectural decision, workflow pattern, or hardware runbook into the persistent WikiSkill repository."""
    try:
        from apex_harness.wikiskill import record_wiki_skill as _rw
        return _rw(title=title, content=content, category=category, tags=tags)
    except Exception as e:
        return f"Error recording wiki skill: {str(e)}"


TOOLS_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "bash_exec": bash_exec,
    "run_tdp_pipeline": run_tdp_pipeline,
    "rag_search": rag_search,
    "rag_ingest": rag_ingest,
    "recall_memory": recall_memory,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_branch": git_branch,
    "git_commit": git_commit,
    "session_log": session_log,
    "consult_wiki": consult_wiki,
    "record_wiki_skill": record_wiki_skill,
}

# ---------------------------------------------------------------------------
# Tiered Tool Disclosure (Jev Engineering pattern)
# ---------------------------------------------------------------------------
#
# O problema: expor todos os schemas completos em cada turno custa 5-12% do
# token budget em overhead fixo -- tokens gastos em ferramentas que o modelo
# nunca vai usar nessa mensagem.
#
# Solução em 3 tiers (Jev Engineering for Coding Agents, 2026):
#
#   TIER 1 · SNIPPETS  → 1 linha por ferramenta, ~200 tokens total
#                         Sempre visível. Serve como "menu" para o modelo.
#
#   TIER 2 · ON-DEMAND → schemas completos das ferramentas seleccionadas
#                         Injectado pelo harness baseado na intenção do turno.
#
#   TIER 3 · DOCS      → descrição longa / exemplos (reservado para futuro)
#
# O `TOOLS_DEFINITION` original continua a existir como fonte de verdade
# para os schemas completos (Tier 2). O `get_tool_schemas_for` extrai os
# schemas apenas das ferramentas necessárias.
# ---------------------------------------------------------------------------

TOOLS_SNIPPETS = """Ferramentas disponíveis (chame com tool_call):
web_search(query) · fetch_url(url) · list_dir(path) · read_file(path, [start_line], [end_line])
write_file(path, content) · edit_file(path, old_text, new_text) · bash_exec(command)
git_status([repo_path]) · git_diff([repo_path], [staged], [file_path]) · git_log([repo_path], [n])
git_branch([repo_path], [create], [switch_to]) · git_commit(message, [repo_path], [add_all])
rag_search(query, [mode], [top_k]) · rag_ingest(path) · recall_memory(query, [top_k])
run_tdp_pipeline(task, [execute_code]) · session_log(event_type, content)
consult_wiki(query, [category]) · record_wiki_skill(title, content, [category], [tags])"""


def get_tool_schemas_for(tool_names: "List[str]", all_tools: "List[Dict[str, Any]]") -> "List[Dict[str, Any]]":
    """Return full OpenAI-format schemas only for the requested tool names (Tier 2)."""
    name_set = set(tool_names)
    return [t for t in all_tools if t["function"]["name"] in name_set]


# ---------------------------------------------------------------------------
# Tool Output Compressor (Visibility Ladder)
# ---------------------------------------------------------------------------
#
# Outputs longos de ferramentas consomem 10-20% do token budget em sessões
# longas porque entram *completos* no histórico e são reenviados a cada turno.
# (grep de 2400 linhas → aparece em todos os turnos seguintes)
#
# Regras:
#   • bash_exec / grep / find: truncar a N linhas, guardar contagem
#   • read_file: truncar a MAX_FILE_LINES linhas
#   • outros: truncar a MAX_GENERIC_CHARS caracteres
#
# O texto truncado inclui um marcador "[X lines hidden — use read_file ou
# bash_exec para ver mais]" para que o modelo saiba que pode pedir mais.
# ---------------------------------------------------------------------------

_OUTPUT_MAX_LINES = {
    "bash_exec":   80,   # comandos: grep/find/ps muito barulhentos
    "read_file":  120,   # ficheiros: já tem paginação nativa
    "git_diff":   200,   # diffs podem ser longos mas são sempre úteis
    "git_log":     40,
    "list_dir":    60,
    "web_search":  60,
    "fetch_url":   80,
}
_OUTPUT_MAX_CHARS_DEFAULT = 6000  # fallback para ferramentas não listadas


def compress_tool_output(tool_name: str, output: str) -> str:
    """
    Compress a tool output before it enters the conversation history.

    Keeps the most informative lines, appending a truncation marker so the
    model knows it can request the rest explicitly.
    """
    if not output:
        return output

    max_lines = _OUTPUT_MAX_LINES.get(tool_name)

    if max_lines is not None:
        lines = output.splitlines()
        if len(lines) <= max_lines:
            return output
        kept = lines[:max_lines]
        hidden = len(lines) - max_lines
        kept.append(
            f"\n… [{hidden} linha(s) omitida(s) para poupar contexto — "
            f"use bash_exec ou read_file para ver mais]"
        )
        return "\n".join(kept)

    # Fallback: character limit
    if len(output) > _OUTPUT_MAX_CHARS_DEFAULT:
        hidden_chars = len(output) - _OUTPUT_MAX_CHARS_DEFAULT
        return (
            output[:_OUTPUT_MAX_CHARS_DEFAULT]
            + f"\n… [{hidden_chars} caractere(s) omitido(s)]"
        )

    return output


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search indexed local documents, codebase, and notes using Hybrid RAG (Vector + BM25) and Rerank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural language query or question."},
                    "mode": {"type": "string", "enum": ["hybrid", "rerank", "dense", "lexical"], "description": "Retrieval mode (default: hybrid)."},
                    "top_k": {"type": "integer", "description": "Number of top chunks to return (default: 5)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_ingest",
            "description": "Ingest a local file or directory into the persistent Vector & Inverted Index store using Ollama nomic-embed-text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file or folder to ingest."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for recent information, documentation, news, or technical solutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Download and read readable text content from a web page URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the page to read."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories within a given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (defaults to current dir '.')."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content with line numbers. Paginates automatically for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "start_line": {"type": "integer", "description": "Starting line number (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Ending line number (inclusive)."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a file. Parent directories are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file to write."},
                    "content": {"type": "string", "description": "Full content to write to the file."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific substring or code block in a file with new code/text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit."},
                    "old_text": {"type": "string", "description": "Exact text segment to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run shell commands in bash on the local machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tdp_pipeline",
            "description": "Run the Task-Decoupled Planning (TDP) pipeline for complex architectural, coding, or multi-step engineering tasks. Uses isolated cognitive stages (Researcher -> Planner -> Executor with self-healing code validation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The detailed problem or requirements description."},
                    "execute_code": {"type": "boolean", "description": "Whether to execute and validate generated code in a sandbox (default: true)."}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Recall past conversation history, notes, decisions, or archived session memory using semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to locate relevant historical context or past discussions."},
                    "top_k": {"type": "integer", "description": "Number of relevant memory snippets to return (default: 5)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working tree status of a git repository (branch and short status).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to git repository (default: current directory)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff between commits, staged changes, or working tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to git repository (default: current directory)."},
                    "staged": {"type": "boolean", "description": "If true, view staged changes (--staged)."},
                    "file_path": {"type": "string", "description": "Optional specific file path to diff."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent commit history in oneline decorated format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to git repository (default: current directory)."},
                    "n": {"type": "integer", "description": "Number of recent commits to display (default: 10)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List, create, or switch git branches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to git repository (default: current directory)."},
                    "create": {"type": "string", "description": "Name of new branch to create and switch to."},
                    "switch_to": {"type": "string", "description": "Name of existing branch to switch to."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Record changes to the git repository with an explicit commit message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to git repository (default: current directory)."},
                    "message": {"type": "string", "description": "Commit message describing the changes."},
                    "add_all": {"type": "boolean", "description": "If true, automatically stage all modified and new files (git add -A) before commit."}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "session_log",
            "description": "Log an architectural decision, touched file, open TODO, or note into persistent session memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "enum": ["decision", "file_touched", "todo_open", "todo_done", "note"], "description": "Category of session event."},
                    "content": {"type": "string", "description": "Text details of the event or decision."}
                },
                "required": ["event_type", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_wiki",
            "description": "Search and read persistent WikiSkill articles, runbooks, architecture patterns, and hardware configurations accumulated from previous agent experiences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or topic to search in the persistent wiki."},
                    "category": {"type": "string", "description": "Optional category filter (e.g., 'hardware', 'workflows', 'architecture', 'gotchas')."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_wiki_skill",
            "description": "Compile and save a new skill card, solution pattern, or hardware runbook into the persistent Wiki knowledge base (arXiv:2608.27454).",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Clear descriptive title of the skill/runbook."},
                    "content": {"type": "string", "description": "Markdown body detailing context, steps, code patterns, and common gotchas."},
                    "category": {"type": "string", "description": "Category: 'workflows', 'hardware', 'architecture', or 'gotchas'."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Search tags for discovery."}
                },
                "required": ["title", "content"]
            }
        }
    }
]

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a registered tool or MCP tool safely and return string output."""
    import time
    t0 = time.perf_counter()
    status = "success"
    result = ""

    if name.startswith("mcp_"):
        try:
            from apex_harness.mcp_client import get_mcp_manager
            result = get_mcp_manager().execute_mcp_tool(name, arguments)
        except Exception as e:
            status = "error"
            result = f"Error executing MCP tool '{name}': {str(e)}"
    elif name not in TOOLS_REGISTRY:
        status = "error"
        result = f"Error: Tool '{name}' is not recognized."
    else:
        try:
            func = TOOLS_REGISTRY[name]
            result = func(**arguments)
        except Exception as e:
            status = "error"
            result = f"Error executing '{name}' with args {arguments}: {str(e)}"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Observability tracing (enabled unless explicitly APEX_TRACE=0)
    if os.environ.get("APEX_TRACE", "1").lower() not in ("0", "false", "no"):
        try:
            from apex_harness.trace import get_trace_logger, TraceEvent
            from datetime import datetime, timezone
            sess_id = os.environ.get("APEX_SESSION_ID", "default_session")
            args_repr = str(arguments)[:200]
            res_repr = str(result)[:300]
            tokens_in = max(1, len(args_repr) // 4)
            tokens_out = max(1, len(res_repr) // 4)
            ev = TraceEvent(
                session_id=sess_id,
                turn=0,
                tool=name,
                args_summary=args_repr,
                result_summary=res_repr,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=os.environ.get("APEX_MODEL", "local"),
                ts=datetime.now(timezone.utc).isoformat(),
                status=status
            )
            get_trace_logger().log_event(ev)
        except Exception:
            pass

    return result
