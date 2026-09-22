"""
Suporte a imagens: deteção do projetor multimodal e construção da mensagem.

O llama-server so ve imagens quando arranca com `-mm/--mmproj FILE`, e responde
a isso no /props:

    modalities: {"vision": false, "video": false, "audio": false}

Sem esse ficheiro o modelo e texto apenas. Medido nesta maquina: ao lado do
Ternary-Bonsai-2-27B esta o Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (601 MB).
"""

import struct
from unittest.mock import MagicMock

import pytest

from apex_harness.core import ApexAgent, message_text
from apex_harness.hwtune import find_mmproj


def _chunk(content=None, reasoning=None):
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.tool_calls = None
    ch = MagicMock()
    ch.choices = [MagicMock(delta=delta)]
    ch.usage = None
    ch.model_extra = {}
    return ch


def _agent(chunks):
    a = ApexAgent.__new__(ApexAgent)
    a.model_name = "teste"
    a.temperature = 0.0
    a.repeat_penalty = 1.0
    a.max_turns = 3
    a.history = [{"role": "system", "content": "sys"}]
    a.tools = []
    a.stable_tools = True
    a._in_reasoning = False
    a.enable_mcp = False
    a.last_usage = {}
    a.session_prompt_tokens = 0
    a.session_completion_tokens = 0
    a.session_cached_tokens = 0
    a._recall_from_rag = lambda _u: ""
    a.auto_compact_if_needed = lambda: None
    a._get_relevant_tools = lambda _u: None
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    a.client = MagicMock()
    a.client.chat.completions.create.return_value = stream
    return a


IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def test_message_with_image_uses_the_multimodal_block_format():
    """
    O llama-server (mtmd) espera `content` como lista de blocos. Sem imagens o
    content tem de continuar a ser string, para nao mexer no que ja funciona.
    """
    a = _agent([_chunk(content="e um gato")])
    a.step("o que e isto?", images=[IMG])
    msg = a.history[-2]  # [-1] e a resposta do assistente
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "o que e isto?"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"] == IMG


def test_message_without_image_stays_a_plain_string():
    a = _agent([_chunk(content="ok")])
    a.step("ola")
    assert isinstance(a.history[-2]["content"], str)


def test_image_only_message_is_allowed():
    """Uma imagem sem texto e uma pergunta legitima."""
    a = _agent([_chunk(content="e um gato")])
    a.step("", images=[IMG])
    msg = a.history[-2]
    assert isinstance(msg["content"], list)
    assert [b["type"] for b in msg["content"]] == ["image_url"]


def test_message_text_flattens_images_without_dumping_base64():
    """
    Contagem de tokens, RAG e compactacao usam `message_text`. `str(lista)`
    daria a representacao Python com o base64 inteiro dentro -- lixo, e enchia o
    contexto com milhares de caracteres que nao sao texto nenhum.
    """
    conteudo = [
        {"type": "text", "text": "descreve isto"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}},
    ]
    texto = message_text(conteudo)
    assert "descreve isto" in texto
    assert "[imagem]" in texto
    assert "AAAA" not in texto
    assert len(texto) < 100
    # E os casos simples continuam a funcionar.
    assert message_text("ola") == "ola"
    assert message_text(None) == ""


def test_find_mmproj_picks_the_projector_beside_the_model(tmp_path):
    (tmp_path / "modelo-Q4_K_M.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "modelo-mmproj-Q8_0.gguf").write_bytes(b"y" * 1024)
    encontrado = find_mmproj(str(tmp_path / "modelo-Q4_K_M.gguf"))
    assert encontrado is not None
    assert "mmproj" in encontrado


def test_find_mmproj_returns_none_when_there_is_no_projector(tmp_path):
    """Ausencia de projetor e o caso normal, nao um erro."""
    (tmp_path / "modelo-Q4_K_M.gguf").write_bytes(b"x" * 1024)
    assert find_mmproj(str(tmp_path / "modelo-Q4_K_M.gguf")) is None
