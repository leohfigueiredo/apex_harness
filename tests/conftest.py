"""
Fixtures partilhadas pelos testes do Apex Harness.

Aqui vive uma regra que impede a classe mais comum de falha nos CI:
os modulos de sessao/RAG gravam em SQLite sob ~/.apex_sessions (e em
~/.apex_sessions/apex_rag.db). Num ambiente com o home em modo de leitura -- um
sandbox, um container, um runner sem permissao de escrita -- o teste rebentava
com

    sqlite3.OperationalError: attempt to write a readonly database

em vez de testar o que devia. Redirecionar o diretorio para um temp por sessao
torna os testes hermeticos: nao dependem de onde o home aponta nem de quem pode
la escrever.
"""

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _hermetic_state_dirs(tmp_path_factory):
    """
    Aponta APEX_SESSION_DIR (SQLite de sessoes) para um diretorio temporario.

    `autouse=True` + `scope=session`: aplica-se a TODOS os testes uma unica vez,
    sem que cada teste tenha de se lembrar de o fazer.
    """
    base = tmp_path_factory.mktemp("apex_state")
    os.environ["APEX_SESSION_DIR"] = str(base)
    # O RAG tambem grava SQLite em ~/.apex_sessions (apex_rag.db). Mesmo problema,
    # mesma correcao: apontar para o temp.
    os.environ["APEX_RAG_DB"] = str(base / "apex_rag.db")
    return base
