"""
A mensagem de erro ao trocar de modelo tem de mostrar o erro REAL, nao o log de
acessos HTTP do servidor a morrer.

Medido no harness: a troca para o Qwen3.6-35B falhou e a mensagem mostrava
`GET /health 200` repetido -- o acesso log do servidor antigo -- em vez de
`unknown model architecture: 'qwen4exp'` ou `Address already in use`.
"""

from apex_harness.server import _extrair_erro_do_log


def _escrever(tmp_path, conteudo):
    p = tmp_path / "log.txt"
    p.write_text(conteudo, encoding="utf-8")
    return str(p)


def test_unknown_architecture_is_translated(tmp_path):
    log = _escrever(tmp_path, (
        "0.00.1 E llama_model_load: error loading model: unknown model architecture: 'qwen4exp'\n"
        "0.00.2 E llama_model_load_from_file_impl: failed to load model\n"
        "0.00.3 E srv llama_server: exiting due to model loading error\n"
    ))
    msg = _extrair_erro_do_log(log, "Qwen3.8-Flash-Next-131B")
    assert "qwen4exp" in msg
    assert "nao suporta" in msg


def test_address_in_use_is_surfaced_not_the_access_log(tmp_path):
    log = _escrever(tmp_path, (
        '127.0.0.1 - - [25/Sep/2026 20:38:53] "GET /health HTTP/1.1" 200 -\n'
        "OSError: [Errno 98] Address already in use\n"
        "srv llama_server: exiting due to HTTP server error\n"
    ))
    msg = _extrair_erro_do_log(log, "Qwen3.6-35B")
    assert "Address already in use" in msg
    # O lixo de acesso log NAO pode aparecer na mensagem.
    assert "GET /health" not in msg


def test_access_log_lines_are_filtered(tmp_path):
    log = _escrever(tmp_path, (
        '127.0.0.1 - - "GET /health HTTP/1.1" 200 -\n'
        '127.0.0.1 - - "GET /v1/models HTTP/1.1" 200 -\n'
    ))
    msg = _extrair_erro_do_log(log, "M")
    assert "GET /health" not in msg


def test_empty_log_gives_a_clean_fallback(tmp_path):
    msg = _extrair_erro_do_log(str(tmp_path / "inexistente.log"), "M")
    assert "sem detalhe" in msg or "Falha ao carregar" in msg
