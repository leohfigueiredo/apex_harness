"""
Porque e que o harness "comecava e depois parava" (diagnosticado em 09/10).

Sessao real com Qwen3.6-35B-A3B: depois de criar 14 ficheiros, o raciocinio do
modelo terminou em
    "Vou fazer tudo de uma vez.\n</parameter>\n</function>\n</tool_call>"
sem `content` nem `tool_calls`. O `step()` tratava isso como resposta final.
Noutro turno, o guarda de ciclo cortava a 3a chamada igual de `bash_exec`, sem
avisar o modelo antes.

Estes testes fixam:
  1. o parser do formato XML nativo do Qwen;
  2. o "nudge" quando a chamada de ferramenta se perde;
  3. o aviso ao modelo antes de abortar um ciclo;
  4. `bash_exec` nao pendura com processos em background.
"""

import time
from unittest.mock import MagicMock, patch

from apex_harness.core import ApexAgent, extract_text_tool_calls, has_tool_markup
from apex_harness.tools import bash_exec


# --------------------------------------------------------------------------- #
# 1. Parser XML do Qwen                                                        #
# --------------------------------------------------------------------------- #

def test_qwen_xml_tool_call_is_parsed():
    texto = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nbackend/app/main.py\n</parameter>\n"
        "<parameter=content>\nimport os\n\n    x = 1\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    chamadas = extract_text_tool_calls(texto)
    assert chamadas == [("write_file", {
        "path": "backend/app/main.py",
        "content": "import os\n\n    x = 1",
    })]


def test_qwen_xml_coerces_numbers_but_not_text_keys():
    texto = (
        "<function=bash_exec><parameter=command>\necho 123\n</parameter>"
        "<parameter=timeout>\n120\n</parameter></function>"
    )
    (nome, args), = extract_text_tool_calls(texto)
    assert nome == "bash_exec"
    assert args == {"command": "echo 123", "timeout": 120}


def test_truncated_qwen_call_is_not_executed():
    """Sem `</function>` a chamada esta incompleta: nao se grava meio ficheiro."""
    texto = "<function=write_file><parameter=path>\nx.py\n</parameter><parameter=content>\nimport"
    assert extract_text_tool_calls(texto) == []
    assert has_tool_markup(texto)


def test_dangling_closing_tags_count_as_markup():
    assert has_tool_markup("Vou fazer tudo.\n</parameter>\n</function>\n</tool_call>")
    assert not has_tool_markup("Resposta normal sem ferramentas.")


# --------------------------------------------------------------------------- #
# Helpers de stream                                                            #
# --------------------------------------------------------------------------- #

def _chunk(content=None, reasoning=None, tool=None):
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    if tool:
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_0"
        tc.function.name = tool[0]
        tc.function.arguments = tool[1]
        delta.tool_calls = [tc]
    else:
        delta.tool_calls = None
    ch = MagicMock()
    ch.choices = [MagicMock(delta=delta)]
    ch.usage = None
    ch.model_extra = {}
    return ch


def _agent(respostas, max_turns=8):
    """`respostas` e uma lista; cada item e a lista de chunks de uma volta."""
    a = ApexAgent.__new__(ApexAgent)
    a.model_name = "teste"
    a.temperature = 0.0
    a.repeat_penalty = 1.0
    a.max_turns = max_turns
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
    a.client = MagicMock()
    fila = list(respostas)

    def _stream(**_kw):
        chunks = fila.pop(0) if fila else [_chunk(content="fim")]
        s = MagicMock()
        s.__iter__ = lambda self: iter(chunks)
        return s

    a.client.chat.completions.create.side_effect = _stream
    return a


_PATCHES = (
    patch("apex_harness.core.execute_tool", return_value="ok"),
    patch("apex_harness.core.compress_tool_output", side_effect=lambda n, r: r),
)


# --------------------------------------------------------------------------- #
# 2. Nudge quando a chamada de ferramenta se perde                              #
# --------------------------------------------------------------------------- #

def test_reasoning_with_broken_tool_markup_triggers_retry_instead_of_stopping():
    a = _agent([
        [_chunk(tool=("write_file", '{"path": "a.py", "content": "x"}'))],
        # O caso real: so raciocinio, com restos de tool call.
        [_chunk(reasoning="Vou fazer tudo de uma vez.\n</parameter>\n</function>\n</tool_call>")],
        [_chunk(tool=("write_file", '{"path": "b.py", "content": "y"}'))],
        [_chunk(content="Backend criado.")],
    ])
    with _PATCHES[0] as ex, _PATCHES[1]:
        out = a.step("faz tudo")
    assert out == "Backend criado."
    assert ex.call_count == 2  # a.py e b.py -- o trabalho continuou
    # A resposta partida nao fica no historico; o aviso sim.
    assert not any("</tool_call>" in str(m.get("content")) for m in a.history)
    assert any("[AVISO DO HARNESS]" in str(m.get("content")) for m in a.history)


def test_qwen_xml_inside_reasoning_is_executed():
    xml = ("<tool_call>\n<function=bash_exec>\n<parameter=command>\nls\n"
           "</parameter>\n</function>\n</tool_call>")
    a = _agent([
        [_chunk(reasoning="Preciso listar.\n" + xml)],
        [_chunk(content="feito")],
    ])
    with _PATCHES[0] as ex, _PATCHES[1]:
        out = a.step("lista")
    assert out == "feito"
    ex.assert_called_once_with("bash_exec", {"command": "ls"})


def test_nudges_are_capped():
    quebrado = [_chunk(reasoning="ups </tool_call>")]
    a = _agent([quebrado, quebrado, quebrado, quebrado])
    with _PATCHES[0], _PATCHES[1]:
        out = a.step("faz")
    # 1 resposta + 2 nudges = 3 pedidos; depois devolve o que ha.
    assert a.client.chat.completions.create.call_count == 3
    assert "<think>" in out


def test_plain_reasoning_answer_on_first_turn_is_not_nudged():
    a = _agent([[_chunk(reasoning="So preciso dizer ok.")]])
    out = a.step("Diga ok")
    assert a.client.chat.completions.create.call_count == 1
    assert "So preciso dizer ok." in out


# --------------------------------------------------------------------------- #
# 3. Ciclo: avisa antes de cortar                                               #
# --------------------------------------------------------------------------- #

def test_repeated_call_gets_warning_and_model_can_recover():
    ls = [_chunk(tool=("bash_exec", '{"command": "ls"}'))]
    a = _agent([ls, ls, [_chunk(tool=("bash_exec", '{"command": "pwd"}'))],
                [_chunk(content="ok")]])
    with _PATCHES[0], _PATCHES[1]:
        out = a.step("faz")
    assert out == "ok"
    tools = [m for m in a.history if m.get("role") == "tool"]
    assert "[AVISO DO HARNESS]" not in tools[0]["content"]
    assert "[AVISO DO HARNESS]" in tools[1]["content"]
    assert "identico" in tools[1]["content"]


# --------------------------------------------------------------------------- #
# 4. bash_exec com processo em background                                       #
# --------------------------------------------------------------------------- #

def test_bash_exec_does_not_hang_on_background_process():
    t0 = time.monotonic()
    out = bash_exec("sleep 30 & echo arrancou", timeout=10)
    assert time.monotonic() - t0 < 5
    assert "arrancou" in out
    assert "Exit Code: 0" in out


def test_bash_exec_timeout_kills_group_and_keeps_partial_output():
    t0 = time.monotonic()
    out = bash_exec("echo parcial; sleep 30 & sleep 30", timeout=1)
    assert time.monotonic() - t0 < 8
    assert "timed out" in out
    assert "parcial" in out


def test_bash_exec_accepts_string_timeout():
    assert "ola" in bash_exec("echo ola", timeout="5")
