"""
Um modelo de raciocinio que termina sem escrever `content` nao pode custar o turno.

Medido nesta maquina (llama-server b10456 + Instinct-Python-Coder-Gemma4-12B, um
modelo de "thinking"): o stream traz o pensamento em `delta.reasoning_content` e a
resposta em `delta.content`. Um pedido de 8 tokens deu
    reasoning_content = 98 chars
    content           =  2 chars ("ok")
Ou seja, sao campos independentes -- e o modelo pode perfeitamente terminar sem
chegar a preencher `content`.

O `step()` so acumulava `delta.content`. Nesse caso o `full_text` ficava vazio, o
raciocinio era deitado fora, o turno morria com "resposta vazia" e nada ficava no
historico. Estes testes fixam o comportamento novo.
"""

from unittest.mock import MagicMock, patch

from apex_harness.core import ApexAgent


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


def _agent_com_stream(chunks):
    agent = ApexAgent.__new__(ApexAgent)
    agent.model_name = "teste"
    agent.temperature = 0.0
    agent.repeat_penalty = 1.0
    agent.max_turns = 5
    agent.history = [{"role": "system", "content": "sys"}]
    agent.tools = []
    agent.stable_tools = True
    agent._in_reasoning = False
    agent.enable_mcp = False
    agent.last_usage = {}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_cached_tokens = 0
    agent._recall_from_rag = lambda _u: ""
    agent.auto_compact_if_needed = lambda: None
    agent._get_relevant_tools = lambda _u: None
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = stream
    return agent


def test_reasoning_only_turn_keeps_the_reasoning_instead_of_erroring():
    agent = _agent_com_stream([
        _chunk(reasoning="O utilizador quer apenas: ok."),
        _chunk(reasoning=" Nao preciso de ferramentas."),
    ])
    out = agent.step("Diga apenas: ok")
    assert "resposta vazia" not in out.lower()
    assert "O utilizador quer apenas: ok." in out
    assert "<think>" in out
    # E fica no historico, para o turno seguinte ter continuidade.
    assert agent.history[-1]["role"] == "assistant"


def test_reply_with_content_is_unchanged():
    """O caminho normal nao pode ganhar o bloco de raciocinio na resposta."""
    agent = _agent_com_stream([
        _chunk(reasoning="A pensar..."),
        _chunk(content="ok"),
    ])
    out = agent.step("Diga apenas: ok")
    assert out == "ok"
    assert "<think>" not in out


def test_truly_empty_turn_still_reports_a_useful_error():
    agent = _agent_com_stream([_chunk()])
    out = agent.step("Diga algo")
    assert "vazia" in out.lower()


def test_repeat_penalty_is_sent_to_the_server():
    """
    Sem penalizacao de repeticao o modelo entra em ciclo.

    Observado em 18/09: temperatura 0.2 (o default do harness) com
    repeat_penalty 1.0 (desligado no servidor), presence/frequency_penalty 0.0.
    A sequencia repetida tem sempre a probabilidade mais alta e nada empurra o
    modelo para fora do ciclo -- o raciocinio repetia "pode ajustar o
    curriculo..." indefinidamente.

    `repeat_penalty` e uma extensao do llama-server, nao faz parte do esquema
    OpenAI, portanto tem de ir em `extra_body`.
    """
    agent = _agent_com_stream([_chunk(content="ok")])
    agent.repeat_penalty = 1.1
    agent.step("Diga apenas: ok")
    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"repeat_penalty": 1.1}


def test_repeat_penalty_is_omitted_when_disabled():
    agent = _agent_com_stream([_chunk(content="ok")])
    agent.repeat_penalty = 1.0
    agent.step("Diga apenas: ok")
    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs
