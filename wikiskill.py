"""
Apex Harness — WikiSkill Engine (wikiskill.py)
Implements 'WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution'
(arXiv:2608.27454).

Co-evolves agent skills with a persistent Markdown knowledge base (wiki),
separating raw execution experience (sessions.db), accumulated knowledge (wiki/),
and executable skills (agent tools & runbooks).
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple


class WikiStore:
    """Manages the persistent Markdown wiki on disk."""

    def __init__(self, wiki_dir: Optional[str] = None):
        if wiki_dir:
            self.wiki_dir = Path(wiki_dir).resolve()
        else:
            # Check local project first (.apex/wiki), then global ~/.apex_sessions/wiki
            local_apex = Path.cwd() / ".apex" / "wiki"
            if local_apex.exists() or (Path.cwd() / ".apex").exists():
                self.wiki_dir = local_apex.resolve()
            else:
                base = os.environ.get("APEX_WIKI_DIR", str(Path.home() / ".apex_sessions" / "wiki"))
                self.wiki_dir = Path(base).resolve()

        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        # Ensure default category subfolders
        for cat in ["hardware", "workflows", "architecture", "gotchas", "general"]:
            (self.wiki_dir / cat).mkdir(parents=True, exist_ok=True)

    def _slugify(self, title: str) -> str:
        slug = re.sub(r'[^\w\s-]', '', title.lower().strip())
        return re.sub(r'[-\s]+', '_', slug)[:60]

    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML-like frontmatter from markdown."""
        meta: Dict[str, Any] = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                raw_meta = parts[1]
                content = parts[2].lstrip()
                for line in raw_meta.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if v.startswith("[") and v.endswith("]"):
                            try:
                                meta[k] = json.loads(v)
                            except Exception:
                                meta[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                        else:
                            meta[k] = v.strip("'\"")
        return meta, content

    def save_article(
        self,
        title: str,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        author: str = "agent"
    ) -> Path:
        """Save or update an article in the wiki."""
        cat_dir = self.wiki_dir / category.lower().strip()
        cat_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self._slugify(title)}.md"
        filepath = cat_dir / filename

        now = datetime.now(timezone.utc).isoformat()
        tags_list = tags or []

        frontmatter = [
            "---",
            f"title: \"{title}\"",
            f"category: \"{category}\"",
            f"tags: {json.dumps(tags_list, ensure_ascii=False)}",
            f"author: \"{author}\"",
            f"updated_at: \"{now}\"",
            "---",
            ""
        ]

        full_doc = "\n".join(frontmatter) + content.strip() + "\n"
        filepath.write_text(full_doc, encoding="utf-8")
        return filepath

    def get_article(self, name_or_title: str) -> Optional[Dict[str, Any]]:
        """Retrieve an article by title or filename slug."""
        target_slug = self._slugify(name_or_title)
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.stem == target_slug:
                try:
                    text = md_file.read_text(encoding="utf-8")
                    meta, body = self._parse_frontmatter(text)
                    return {
                        "path": str(md_file),
                        "slug": md_file.stem,
                        "title": meta.get("title", md_file.stem),
                        "category": meta.get("category", md_file.parent.name),
                        "tags": meta.get("tags", []),
                        "updated_at": meta.get("updated_at", ""),
                        "content": body
                    }
                except Exception:
                    pass
        return None

    def list_articles(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all articles in the wiki."""
        articles = []
        for md_file in sorted(self.wiki_dir.rglob("*.md")):
            if md_file.name.startswith("."):
                continue
            cat = md_file.parent.name
            if category and cat.lower() != category.lower():
                continue
            try:
                text = md_file.read_text(encoding="utf-8")
                meta, body = self._parse_frontmatter(text)
                title = meta.get("title", md_file.stem.replace("_", " ").title())
                snippet = body.strip().splitlines()[0] if body.strip() else ""
                articles.append({
                    "path": str(md_file),
                    "slug": md_file.stem,
                    "title": title,
                    "category": meta.get("category", cat),
                    "tags": meta.get("tags", []),
                    "updated_at": meta.get("updated_at", ""),
                    "snippet": snippet[:140]
                })
            except Exception:
                continue
        return articles

    def search_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search articles by keyword or tag relevance."""
        q_tokens = set(re.findall(r'\w+', query.lower()))
        if not q_tokens:
            return self.list_articles()[:top_k]

        scored = []
        for art in self.list_articles():
            score = 0
            title_tokens = set(re.findall(r'\w+', art["title"].lower()))
            tag_tokens = {t.lower() for t in art.get("tags", [])}
            
            # Match title
            score += len(q_tokens & title_tokens) * 3
            # Match tags
            score += len(q_tokens & tag_tokens) * 2
            # Match snippet
            snippet_tokens = set(re.findall(r'\w+', art.get("snippet", "").lower()))
            score += len(q_tokens & snippet_tokens)

            if score > 0:
                full_art = self.get_article(art["slug"])
                content = full_art["content"] if full_art else art["snippet"]
                scored.append((score, {**art, "content": content}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


class WikiSkillCompiler:
    """
    Compiles raw execution experience from session memory into persistent
    knowledge cards following the WikiSkill framework.
    """

    def __init__(self, store: Optional[WikiStore] = None):
        self.store = store or WikiStore()

    def compile_from_sessions(self, limit_sessions: int = 5) -> Dict[str, Any]:
        """Read recent sessions from session memory and synthesize wiki articles."""
        from apex_harness.session_memory import get_session_memory
        mem = get_session_memory()
        sessions = mem.list_sessions(limit=limit_sessions)

        compiled_count = 0
        articles_created = []

        for sess in sessions:
            sid = sess["id"]
            events = mem.get_events(sid)
            if not events:
                continue

            decisions = [e["content"] for e in events if e["event_type"] == "decision"]
            files = sorted(list({e["content"] for e in events if e["event_type"] == "file_touched"}))
            done_todos = [e["content"] for e in events if e["event_type"] == "todo_done"]
            notes = [e["content"] for e in events if e["event_type"] == "note"]

            # Only compile sessions that have substantial insights
            if not decisions and not done_todos and not notes and not files:
                continue

            summary = sess.get("summary") or f"Sessão {sid[:8]}"
            title = f"Skill: {summary[:50]}"
            
            # Format Markdown Wiki Article
            lines = [
                f"# {title}",
                "",
                f"> **Compilado de:** Sessão `{sid}` | **Modelo:** `{sess.get('model') or 'local'}`",
                "",
                "## 🎯 Contexto e Objetivo",
                f"{sess.get('summary') or 'Tarefas executadas durante a sessão.'}",
                ""
            ]

            if decisions:
                lines.extend([
                    "## ⚖️ Decisões Arquiteturais e Lições Aprendidas",
                    *[f"- {d}" for d in decisions],
                    ""
                ])

            if done_todos:
                lines.extend([
                    "## ✅ Padrões e Tarefas Resolvidas",
                    *[f"- {t}" for t in done_todos],
                    ""
                ])

            if files:
                lines.extend([
                    "## 📁 Arquivos Relevantes e Componentes",
                    *[f"- `{f}`" for f in files[:10]],
                    ""
                ])

            if notes:
                lines.extend([
                    "## 📝 Notas Técnicas e Gotchas",
                    *[f"- {n}" for n in notes],
                    ""
                ])

            content = "\n".join(lines)
            tags = ["compiled-experience", "session-history"]
            if sess.get("model"):
                tags.append(self.store._slugify(sess["model"]))

            path = self.store.save_article(
                title=title,
                content=content,
                category="workflows",
                tags=tags,
                author="wikiskill-compiler"
            )
            compiled_count += 1
            articles_created.append(str(path.name))

        return {
            "status": "ok",
            "compiled_count": compiled_count,
            "articles": articles_created
        }


# Global helpers for tools and server
_GLOBAL_WIKI: Optional[WikiStore] = None

def get_wiki_store() -> WikiStore:
    global _GLOBAL_WIKI
    if _GLOBAL_WIKI is None:
        _GLOBAL_WIKI = WikiStore()
    return _GLOBAL_WIKI


def consult_wiki(query: str, category: Optional[str] = None) -> str:
    """
    Search and read articles from the persistent WikiSkill knowledge base.
    Use this to recall hardware configurations, architecture patterns, or previous runbooks.
    """
    store = get_wiki_store()
    hits = store.search_articles(query, top_k=3)
    if not hits:
        all_arts = store.list_articles(category=category)
        if not all_arts:
            return f"A Wiki de conhecimento está vazia no momento (pasta {store.wiki_dir})."
        titles = [f"- [{a['category']}] {a['title']}" for a in all_arts[:10]]
        return f"Nenhum artigo encontrado para '{query}'. Artigos disponíveis:\n" + "\n".join(titles)

    output = [f"=== WIKISKILL: Encontrados {len(hits)} artigo(s) para '{query}' ==="]
    for h in hits:
        output.append(f"\n--- Artigo: {h['title']} (Categoria: {h['category']}) ---")
        output.append(h.get("content", h.get("snippet", "")))
    return "\n".join(output)


def record_wiki_skill(title: str, content: str, category: str = "workflows", tags: Optional[List[str]] = None) -> str:
    """
    Record a new skill, architecture pattern, or hardware runbook into the persistent Wiki.
    Allows future agent sessions to immediately benefit from this knowledge.
    """
    store = get_wiki_store()
    path = store.save_article(title=title, content=content, category=category, tags=tags or [])
    return f"Artigo salvo com sucesso na Wiki: '{title}' em {path.name}"
