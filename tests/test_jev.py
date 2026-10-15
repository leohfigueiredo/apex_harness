"""
Unit tests for Jev Decider (System 1 layer)
"""

import unittest
from unittest.mock import patch

from apex_harness.jev_decider import get_jev, JevChoice, JevNoul, JevPermit


def _jeu_resposta(json_str: str):
    """_jev_call devolve (raw, backend, latency_ms)."""
    return (json_str, "mock-backend", 1.0)


class TestJevDecider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jev = get_jev()

    def test_ask_choice(self):
        # Antes ia ao LLM REAL (portas 8090/8091/1234): sem backend, o fallback
        # escolhia a primeira opcao e o teste passava por acaso. Agora o
        # _jev_call e mockado -- o teste mede a PARSE da resposta, nao a
        # disponibilidade da rede.
        with patch("apex_harness.jev_decider._jev_call",
                   return_value=_jeu_resposta('{"type": "choice", "selected": "ls", "confidence": 0.9}')):
            fut = self.jev.ask_choice(
                state="Usuário solicitou listar arquivos.",
                question="Qual comando executar?",
                options=["ls", "rm", "cat"]
            )
            res = fut.result(timeout=10)
        self.assertIsInstance(res, JevChoice)
        self.assertEqual(res.selected, "ls")
        self.assertTrue(res.ok)

    def test_ask_noul(self):
        with patch("apex_harness.jev_decider._jev_call",
                   return_value=_jeu_resposta('{"type": "noul", "probability_yes": 0.95, "confidence": 0.9}')):
            fut = self.jev.ask_noul(
                state="Comando: rm -rf /",
                question="Esse comando é perigoso?"
            )
            res = fut.result(timeout=10)
        self.assertIsInstance(res, JevNoul)
        self.assertTrue(res.is_yes)

    def test_ask_permit(self):
        with patch("apex_harness.jev_decider._jev_call",
                   return_value=_jeu_resposta('{"type": "choice", "selected": "deny", "confidence": 0.9}')):
            fut = self.jev.ask_permit("rm -rf /")
            res = fut.result(timeout=10)
        self.assertIsInstance(res, JevPermit)
        self.assertEqual(res.verdict, "deny")

    def test_clef_and_npu_endpoints(self):
        from apex_harness.jev_decider import get_jev_endpoints
        endpoints = get_jev_endpoints()
        names = [ep["name"] for ep in endpoints]
        self.assertTrue(any("clef" in n.lower() for n in names))
        self.assertTrue(any("primary" in n.lower() or "8000" in ep["api"] or "fallback" in n.lower() for n, ep in [(e["name"], e) for e in endpoints]))

    def test_hardware_tracking(self):
        from apex_harness.jev_decider import _detect_hardware_label
        self.assertIn("NPU", _detect_hardware_label("NPU-XDNA2-Lemonade-8090"))
        self.assertIn("Clef", _detect_hardware_label("llama-server-clef-8091"))
        self.assertIn("Ollama", _detect_hardware_label("Ollama-clef-flash"))


    def test_decisionforge_systemone_endpoint(self):
        from apex_harness.jev_decider import get_jev_endpoints, _detect_hardware_label
        endpoints = get_jev_endpoints()
        self.assertTrue(any(ep.get("format") == "systemone" for ep in endpoints))
        self.assertIn("Clef Joint Schema Head", _detect_hardware_label("DecisionForge-System1-8501"))


if __name__ == "__main__":
    unittest.main()
