"""
Unit tests for Jev Decider (System 1 layer)
"""

import unittest
from apex_harness.jev_decider import get_jev, JevChoice, JevScore, JevNoul, JevPermit


class TestJevDecider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jev = get_jev()

    def test_ask_choice(self):
        fut = self.jev.ask_choice(
            state="Usuário solicitou listar arquivos.",
            question="Qual comando executar?",
            options=["ls", "rm", "cat"]
        )
        res = fut.result(timeout=10)
        self.assertIsInstance(res, JevChoice)
        self.assertIn(res.selected, ["ls", "rm", "cat"])

    def test_ask_noul(self):
        fut = self.jev.ask_noul(
            state="Comando: rm -rf /",
            question="Esse comando é perigoso?"
        )
        res = fut.result(timeout=10)
        self.assertIsInstance(res, JevNoul)
        self.assertTrue(res.is_yes)

    def test_ask_permit(self):
        fut = self.jev.ask_permit("rm -rf /")
        res = fut.result(timeout=10)
        self.assertIsInstance(res, JevPermit)
        self.assertEqual(res.verdict, "deny")


if __name__ == "__main__":
    unittest.main()
