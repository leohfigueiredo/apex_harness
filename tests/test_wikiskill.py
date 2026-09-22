import tempfile
from pathlib import Path
from apex_harness.wikiskill import WikiStore, WikiSkillCompiler, consult_wiki, record_wiki_skill
from apex_harness.session_memory import SessionMemory


def test_wikistore_save_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = WikiStore(wiki_dir=tmpdir)
        
        # Save a new skill article
        saved_path = store.save_article(
            title="Radeon 890M Vulkan Tuning",
            content="## Problem\nK-Quants slow fallback.\n## Solution\nUse -t 12 and verify layer offload.",
            category="hardware",
            tags=["vulkan", "amd", "gfx1103"]
        )
        assert saved_path.exists()
        assert "radeon_890m_vulkan_tuning" in saved_path.name

        # Retrieve by slug
        art = store.get_article("radeon_890m_vulkan_tuning")
        assert art is not None
        assert art["title"] == "Radeon 890M Vulkan Tuning"
        assert art["category"] == "hardware"
        assert "vulkan" in art["tags"]
        assert "K-Quants slow fallback" in art["content"]

        # List articles
        articles = store.list_articles()
        assert len(articles) == 1
        assert articles[0]["title"] == "Radeon 890M Vulkan Tuning"

        # Search articles
        hits = store.search_articles("vulkan tuning")
        assert len(hits) >= 1
        assert hits[0]["title"] == "Radeon 890M Vulkan Tuning"


def test_wikiskill_compiler():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_sessions.db")
        wiki_dir = str(Path(tmpdir) / "wiki")

        # Populate session memory with test events
        mem = SessionMemory(db_path=db_path)
        sid = "test_sess_001"
        mem.open_session(sid, model="qwen3-coder-30b")
        mem.update_summary(sid, "Implementacao do modulo de telemetria NPU")
        mem.log_decision(sid, "Optamos por utilizar /dev/accel/accel0 para NPU Phoenix/Strix.")
        mem.log_file_touched(sid, "/tmp/npu_backend.py")
        mem.add_note(sid, "Requer permissões no grupo render para acesso ao device node.")
        tid = mem.add_todo(sid, "Adicionar checagem de timeout no driver XDNA")
        mem.close_todo(sid, tid)

        # Compile session experience into wiki
        store = WikiStore(wiki_dir=wiki_dir)
        compiler = WikiSkillCompiler(store=store)
        
        # Patch session_memory singleton
        import apex_harness.session_memory as sm
        old_inst = sm._GLOBAL_SESSION_MEMORY
        sm._GLOBAL_SESSION_MEMORY = mem

        try:
            res = compiler.compile_from_sessions(limit_sessions=5)
            assert res["status"] == "ok"
            assert res["compiled_count"] >= 1

            arts = store.list_articles(category="workflows")
            assert len(arts) >= 1
            compiled_art = store.get_article(arts[0]["slug"])
            assert compiled_art is not None
            assert "/dev/accel/accel0" in compiled_art["content"]
            assert "XDNA" in compiled_art["content"]
        finally:
            sm._GLOBAL_SESSION_MEMORY = old_inst


def test_consult_and_record_wiki():
    with tempfile.TemporaryDirectory() as tmpdir:
        import apex_harness.wikiskill as ws
        old_wiki = ws._GLOBAL_WIKI
        ws._GLOBAL_WIKI = WikiStore(wiki_dir=tmpdir)
        try:
            # Record a skill
            res_rec = record_wiki_skill(
                title="Git Fast-Forward Workflow",
                content="Use git merge --ff-only to ensure clean linear history.",
                category="workflows",
                tags=["git", "ci"]
            )
            assert "salvo com sucesso" in res_rec.lower()

            # Consult the wiki
            consult_res = consult_wiki("git linear history")
            assert "WIKISKILL" in consult_res
            assert "Git Fast-Forward Workflow" in consult_res
            assert "--ff-only" in consult_res

            # Empty consult query
            consult_empty = consult_wiki("nonexistent topic 12345")
            assert "Nenhum artigo encontrado" in consult_empty or "Artigos disponíveis" in consult_empty
        finally:
            ws._GLOBAL_WIKI = old_wiki
