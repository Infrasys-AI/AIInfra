"""CPU-only contract tests; fake client outputs are not performance measurements.

SPDX-License-Identifier: MIT
"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import demo


def observation(**changes):
    row = dict(id=1, model="test-model", hardware="test-gpu", framework="test",
               precision="fp8", spec_method="none", offload_mode="off",
               disagg=False, date="2026-10-30", image="image@sha256:example",
               benchmark_type="single_turn", isl=1024, osl=128,
               num_prefill_gpu=8, num_decode_gpu=8, conc=4,
               run_url="https://github.com/example/actions/runs/1",
               metrics=dict(output_tput_tps=800, input_tput_tps=6400,
                            p90_itl=0.02, p99_itl=0.04, median_ttft=0.1))
    row.update(changes)
    return row


class CalculationTests(unittest.TestCase):
    def test_aggregate_count_is_not_doubled_and_time_is_converted(self):
        p = demo.public_point(observation())
        self.assertEqual(p["num_gpus"], 8)
        self.assertEqual(p["output_tps_per_gpu"], 100)
        self.assertEqual(p["total_tps_per_gpu"], 900)
        self.assertEqual(p["p90_itl_ms"], 20)
        self.assertEqual(p["median_ttft_ms"], 100)

    def test_pd_cost_includes_both_pools(self):
        p = demo.public_point(observation(disagg=True, num_prefill_gpu=6, num_decode_gpu=2))
        result = demo.analyze([p], 50, "output", {"test-gpu": 2}, 0.5)[0]
        self.assertEqual(p["num_gpus"], 8)
        self.assertAlmostEqual(result["usd_per_million_tokens"], 2e6 / (3600 * 100 * 0.5))

    def test_absent_price_and_unreachable_target_stay_unknown(self):
        p = demo.public_point(observation())
        self.assertIsNone(demo.analyze([p], 50, "output", {}, 1)[0]["usd_per_million_tokens"])
        result = demo.analyze([p], 51, "output", {"test-gpu": 2}, 1)[0]
        self.assertEqual(result["status"], "no eligible observation")
        self.assertIsNone(result["throughput_tps_per_gpu"])
        self.assertIsNone(result["usd_per_million_tokens"])

    def test_missing_itl_does_not_become_zero_latency(self):
        row = observation(metrics=dict(output_tput_tps=800, input_tput_tps=6400))
        point = demo.public_point(row)
        result = demo.analyze([point], 1, "output", {}, 1)[0]
        self.assertEqual(result["status"], "no eligible observation")

    def test_series_do_not_mix_workloads_precisions_or_dates(self):
        points = [demo.public_point(observation()),
                  demo.public_point(observation(precision="fp4")),
                  demo.public_point(observation(benchmark_type="agentic_traces", isl=None, osl=None)),
                  demo.public_point(observation(date="2026-10-29"))]
        self.assertEqual(len(demo.analyze(points, 50, "total", {}, 1)), 4)

    def test_failed_point_cannot_win_and_total_token_cost_uses_total_throughput(self):
        p = demo.public_point(observation())
        p["failed"] = 1
        self.assertEqual(demo.analyze([p], 50, "output", {}, 1)[0]["status"], "no eligible observation")
        p["failed"] = 0
        out = demo.analyze([p], 50, "output", {"test-gpu": 2}, 1)[0]
        total = demo.analyze([p], 50, "total", {"test-gpu": 2}, 1)[0]
        self.assertAlmostEqual(out["usd_per_million_tokens"] / total["usd_per_million_tokens"], 9)

    def test_invalid_normalization_and_nonfinite_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            demo.public_point(observation(num_prefill_gpu=4, num_decode_gpu=8))
        with self.assertRaises(ValueError):
            demo.public_point(observation(disagg=True, num_prefill_gpu=0))
        for price in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                demo.analyze([demo.public_point(observation())], 50, "output", {"test-gpu": price}, 1)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def call(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return demo.main(list(args))

    def bench_args(self, output):
        return ["bench", "--base-url", "http://127.0.0.1:8000", "--model", "test-model",
                "--hardware", "test-gpu", "--framework", "vllm", "--precision", "fp8",
                "--num-gpus", "2", "--server-note", "synthetic client fixture",
                "--concurrency", "1,4", "--num-prompts", "8", "--output-dir", str(output)]

    def test_sample_and_reanalysis_are_offline_and_no_overwrite(self):
        output = self.root / "sample"
        with patch("demo.urlopen", side_effect=AssertionError("network not allowed")):
            self.assertEqual(self.call("sample", "--output-dir", str(output)), 0)
        self.assertEqual(self.call("sample", "--output-dir", str(output)), 1)
        self.assertEqual(self.call("report", "--input", str(output / "points.json"),
                                   "--output-dir", str(self.root / "again"), "--target", "10000"), 0)
        rows = json.loads((self.root / "again" / "summary.json").read_text())
        self.assertTrue(all(r["point_id"] is None for r in rows))

    def test_dry_run_does_not_launch_client_or_contact_server(self):
        output = self.root / "dry"
        with patch("demo.subprocess.run", side_effect=AssertionError("must not run")):
            self.assertEqual(self.call(*self.bench_args(output), "--dry-run"), 0)
        meta = json.loads((output / "metadata.json").read_text())
        self.assertEqual(meta["status"], "dry-run")
        self.assertIn("--max-concurrency", meta["commands"][1])
        self.assertFalse((output / "report.html").exists())

    def test_real_process_contract_and_failed_requests(self):
        # This fake executable verifies process arguments and artifact handling only.
        client = self.root / "fake-vllm"
        client.write_text('''#!/usr/bin/env python3
import sys,json
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print("FAKE CLIENT - not a benchmark");sys.exit(0)
def arg(name): return sys.argv[sys.argv.index(name)+1]
failed = 1 if "failed" in arg("--result-dir") else 0
data = dict(duration=2,completed=8-failed,failed=failed,
            total_input_tokens=8192,total_output_tokens=1024,
            p90_itl_ms=10,p99_itl_ms=20,median_ttft_ms=100,request_goodput=3)
(Path(arg("--result-dir"))/arg("--result-filename")).write_text(json.dumps(data))
''')
        client.chmod(0o755)
        output = self.root / "ok"
        self.assertEqual(self.call(*self.bench_args(output), "--client", str(client)), 0)
        bundle = json.loads((output / "points.json").read_text())
        self.assertEqual(len(bundle["points"]), 2)
        self.assertEqual(bundle["points"][0]["output_tps_per_gpu"], 256)
        self.assertTrue((output / "client-conc-4.log").exists())
        bad = self.root / "failed"
        self.assertEqual(self.call(*self.bench_args(bad), "--client", str(client)), 1)
        self.assertEqual(json.loads((bad / "metadata.json").read_text())["status"], "failed")
        self.assertTrue((bad / "raw-conc-1.json").exists())
        self.assertFalse((bad / "report.html").exists())

    def test_script_end_tags_are_escaped(self):
        point = demo.public_point(observation(model="</script><script>alert(1)</script>"))
        args = argparse.Namespace(prices=None, target=50, token_type="output", utilization=1)
        with contextlib.redirect_stdout(io.StringIO()):
            demo.finish(self.root, [point], {}, args)
        html = (self.root / "report.html").read_text()
        self.assertNotIn("</script><script>alert(1)", html)
        self.assertIn("\\u003c/script\\u003e", html)


if __name__ == "__main__":
    unittest.main()
