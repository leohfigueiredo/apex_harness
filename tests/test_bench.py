"""
Tests for benchmark_critic in apex_harness.bench
"""

from unittest.mock import patch
from apex_harness.bench import benchmark_critic
from apex_harness.critic import CriticResult
from apex_harness.npu_detect import NPUStatus


def test_benchmark_critic_runs_successfully():
    with patch("apex_harness.critic.urllib.request.urlopen") as mock_url:
        mock_resp = mock_url.return_value.__enter__.return_value
        mock_resp.read.return_value = b'{"choices": [{"message": {"content": "PASS"}}]}'
        mock_resp.status = 200

        res = benchmark_critic(runs=2, verbose=False)

        assert res["runs"] == 2
        assert "default_endpoint" in res
        assert "npu_endpoint" in res
        assert "overhead_analysis" in res
        assert res["overhead_analysis"]["daemon_recommended"] is True
        assert len(res["default_endpoint"]["latencies_ms"]) == 2
        assert len(res["npu_endpoint"]["latencies_ms"]) == 2
