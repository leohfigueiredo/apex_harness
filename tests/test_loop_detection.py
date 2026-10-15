"""
Porque e que o harness "fica parado" no meio do trabalho: o modelo entra num
ciclo de ferramentas, e a interface nao mostrava nada disso.

O `step()` faz ate `max_turns` voltas. Se o modelo pedir a MESMA ferramenta com
os MESMOS argumentos em voltas seguidas, e um loop degenerativo -- queimar as 40
voltas em silencio nao avanca nada. Estes testes fixam a detecao e o facto de o
`on_turn` dar visibilidade a quem manda o pedido.
"""

from unittest.mock import MagicMock, patch

from apex_harness.core import ApexAgent


def _tool_chunk(name, args_json, idx=0):
    tc = MagicMock()
    tc.index = idx
    tc.id = "call_0"
    tc.function.name = name
    tc.function.arguments = args_json
    delta = MagicMock()
    delta.content = None
    delta.reasoning_content = None
    delta.tool_calls = [tc]
    ch = MagicMock()
    ch.choices = [MagicMock(delta=delta)]
    ch.usage = None
    ch.model_extra = {}
    return ch


def _agent_com_sequencia(seq_de_ferramentas):
    """Cada volta devolve UMA ferramenta; a sequencia dita qual."""
    a = ApexAgent.__new__(ApexAgent)
    a.model_name = "teste"
    a.temperature = 0.0
    a.repeat_penalty = 1.0
    a.max_turns = 6
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

    seq = list(seq_de_ferramentas)
    n = {"v": 0}

    def _stream():
        if n["v"] < len(seq):
            nome, args = seq[n["v"]]
            n["v"] += 1
            chunks = [_tool_chunk(nome, args)]
        else:
            # Sem mais ferramentas: devolve um texto final para fechar o turno.
            chunks = [_chunk_texto("feito")]
        s = MagicMock()
        s.__iter__ = lambda self: iter(chunks)
        return s

    a.client.chat.completions.create.side_effect = lambda **kw: _stream()
    return a


def _chunk_texto(conteudo):
    delta = MagicMock()
    delta.content = conteudo
    delta.reasoning_content = None
    delta.tool_calls = None
    ch = MagicMock()
    ch.choices = [MagicMock(delta=delta)]
    ch.usage = None
    ch.model_extra = {}
    return ch


def test_same_tool_three_times_in_a_row_is_detected_as_a_loop():
    a = _agent_com_sequencia([("bash_exec", '{"command": "ls"}')] * 5)
    with patch("apex_harness.core.execute_tool", return_value="ok"), \
         patch("apex_harness.core.compress_tool_output", side_effect=lambda n, r: r):
        out = a.step("faz algo")
    assert "Loop detetado" in out
    assert "bash_exec" in out


def test_different_tools_do_not_trigger_the_loop_guard():
    a = _agent_com_sequencia([
        ("bash_exec", '{"command": "ls"}'),
        ("read_file", '{"path": "x"}'),
        ("bash_exec", '{"command": "pwd"}'),
    ])
    with patch("apex_harness.core.execute_tool", return_value="ok"), \
         patch("apex_harness.core.compress_tool_output", side_effect=lambda n, r: r):
        out = a.step("faz algo")
    assert "Loop detetado" not in out
    assert out == "feito"


def test_on_turn_reports_each_round_for_visibility():
    """Quem manda o pedido (a web) tem de saber em que volta estamos."""
    voltas = []

    a = _agent_com_sequencia([("bash_exec", '{"command": "ls"}')] * 5)
    with patch("apex_harness.core.execute_tool", return_value="ok"), \
         patch("apex_harness.core.compress_tool_output", side_effect=lambda n, r: r):
        a.step("faz algo", on_turn=lambda n: voltas.append(n))
    # 4 voltas ate o loop ser detetado (avisa a partir da 2a, corta na 4a).
    assert voltas == [1, 2, 3, 4]


def test_max_turns_message_uses_the_actual_limit():
    """A mensagem final dizia '15 turnos' quando o limite ja era 40.

    Para chegar ao fim do while (sem loop detetado), o modelo pede uma
    ferramenta DIFERENTE em cada volta, ate esgotar max_turns=6.
    """
    ferramentas = [(f"tool_{i}", f'{{"n": {i}}}') for i in range(6)]
    a = _agent_com_sequencia(ferramentas)
    with patch("apex_harness.core.execute_tool", return_value="ok"), \
         patch("apex_harness.core.compress_tool_output", side_effect=lambda n, r: r):
        out = a.step("faz algo")
    assert "Loop detetado" not in out
    assert "6 voltas" in out and "15" not in out
