"""
O parser de tool calls em texto tem de reconhecer o formato NATIVO dos modelos
que emitem "channels", e nao so o JSON do OpenAI.

Medido no harness, com o Nemotron-30B-A3B: o modelo respondia a primeira
pergunta, escrevia

    <|tool_call>call:bash_exec{command: "ls -R ... | wc -l"}}<tool_call|>

e o turno PARAVA ali -- porque nenhum padrao casava com este formato, o tool
call era tratado como texto, e nenhuma ferramenta executava. Estes testes fixam
os tres formatos que o parser tem de aceitar.
"""

from apex_harness.core import extract_text_tool_calls


def test_nemotron_channel_tool_call():
    texto = (
        '<|channel>thought\n'
        'I will use bash_exec to list the files.<channel|>'
        '<|tool_call>call:bash_exec{command: "ls -R /home/leonardo/Documents/Applications '
        '| grep -E \'(md|pdf|docx|html)$\' | wc -l"}}<tool_call|>'
    )
    chamadas = extract_text_tool_calls(texto)
    assert chamadas, "o formato Nemotron nao foi reconhecido"
    nome, args = chamadas[0]
    assert nome == "bash_exec"
    assert args["command"].startswith("ls -R /home/leonardo/Documents/Applications")


def test_nemotron_multiple_args_lenient_keys():
    # Variação: duas chaves sem aspas e valores numéricos.
    texto = '<|tool_call>call:write_file{path: "x.py", mode: "w"}}<tool_call|>'
    chamadas = extract_text_tool_calls(texto)
    assert len(chamadas) == 1
    nome, args = chamadas[0]
    assert nome == "write_file"
    assert args == {"path": "x.py", "mode": "w"}


def test_standard_json_tag_still_works():
    texto = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
    chamadas = extract_text_tool_calls(texto)
    assert chamadas == [("read_file", {"path": "x.py"})]


def test_code_fence_json_still_works():
    texto = '```tool_call\n{"name": "bash_exec", "arguments": {"command": "ls"}}\n```'
    chamadas = extract_text_tool_calls(texto)
    assert chamadas == [("bash_exec", {"command": "ls"})]


def test_plain_text_without_tool_call_returns_nothing():
    assert extract_text_tool_calls("quantos curriculos tem aqui?") == []
