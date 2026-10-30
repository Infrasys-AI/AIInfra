#!/usr/bin/env python3
"""Inference benchmarking teaching demos. Python 3.10+, standard library only.

SPDX-License-Identifier: MIT
"""

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

API = "https://inferencex.semianalysis.com/api/v1/benchmarks"
HERE = Path(__file__).resolve().parent
SCOPE = ("source", "model", "workload", "hardware", "framework", "precision",
         "spec_method", "offload_mode", "disagg", "date", "image")


def number(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return value


def positive_int(value):
    try:
        result = int(value)
        if result <= 0:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc


def optional_metric(metrics, name, scale=1):
    value = metrics.get(name)
    return None if value is None else number(value, name, zero=True) * scale


def gpu_count(row):
    """Aggregate engines mirror their count; PD engines use both GPU pools."""
    p, d = row.get("num_prefill_gpu"), row.get("num_decode_gpu")
    for value in (p, d):
        number(value, "physical GPU count", zero=True)
        if int(value) != value:
            raise ValueError("physical GPU counts must be integers")
    if row.get("disagg") is True:
        if not p or not d:
            raise ValueError("PD measurements require both physical GPU pool counts")
        return int(number(p + d, "PD GPU count"))
    if p and d and p != d:
        raise ValueError("unequal aggregate role counts: cannot infer GPU count")
    return int(number(max(p, d), "aggregate GPU count"))


def public_point(row):
    metrics = row["metrics"]
    n = gpu_count(row)
    workload = row.get("benchmark_type")
    if workload != "agentic_traces":
        if row.get("isl") is None or row.get("osl") is None:
            raise ValueError("fixed workload requires ISL and OSL")
        workload = f"{workload}:{row['isl']}/{row['osl']}"
    point = {key: row.get(key, "unknown") for key in SCOPE}
    point.update(
        source="InferenceX public observation", workload=workload,
        id=str(row["id"]), num_gpus=n, concurrency=row.get("conc"),
        output_tps_per_gpu=number(metrics.get("output_tput_tps"), "output TPS") / n,
        total_tps_per_gpu=(number(metrics.get("input_tput_tps"), "input TPS", zero=True)
                           + number(metrics.get("output_tput_tps"), "output TPS")) / n,
        p90_itl_ms=optional_metric(metrics, "p90_itl", 1000),
        p99_itl_ms=optional_metric(metrics, "p99_itl", 1000),
        median_ttft_ms=optional_metric(metrics, "median_ttft", 1000),
        completed=metrics.get("total_requests_completed"), failed=None,
        run_url=row.get("run_url"),
        topology={key: row.get(key) for key in (
            "prefill_tp", "decode_tp", "prefill_ep", "decode_ep",
            "prefill_num_workers", "decode_num_workers")},
    )
    return point


def local_point(result, args, concurrency, version, date):
    n = args.num_gpus
    completed = number(result.get("completed"), "successful requests", zero=True)
    failed = number(result.get("failed"), "failed requests", zero=True)
    if completed + failed != args.num_prompts:
        raise ValueError("result request counts do not match --num-prompts")
    duration = number(result.get("duration"), "duration")
    output = number(result.get("total_output_tokens"), "output tokens", zero=True)
    inputs = number(result.get("total_input_tokens"), "input tokens", zero=True)
    return dict(
        source="Local unofficial measurement", id=f"conc-{concurrency}",
        model=args.model, workload=f"single_turn:{args.isl}/{args.osl}",
        hardware=args.hardware, framework=args.framework, precision=args.precision,
        spec_method=args.spec_method, offload_mode="user-described",
        disagg=args.disagg, date=date, image=args.server_note,
        num_gpus=n, concurrency=concurrency,
        output_tps_per_gpu=output / duration / n,
        total_tps_per_gpu=(inputs + output) / duration / n,
        p90_itl_ms=optional_metric(result, "p90_itl_ms"),
        p99_itl_ms=optional_metric(result, "p99_itl_ms"),
        median_ttft_ms=optional_metric(result, "median_ttft_ms"),
        completed=completed, failed=failed, run_url=None,
        request_goodput=result.get("request_goodput"), client_version=version,
    )


def analyze(points, target, token_type, prices, utilization):
    """Choose the highest-throughput eligible observation in each series."""
    number(target, "target")
    number(utilization, "utilization")
    if utilization > 1:
        raise ValueError("utilization must be <= 1")
    for hardware, price in prices.items():
        number(price, f"price for {hardware}")
    groups = {}
    for point in points:
        groups.setdefault(tuple(point.get(k) for k in SCOPE), []).append(point)
    results = []
    for key, group in groups.items():
        eligible = [p for p in group if p.get("p90_itl_ms") is not None
                    and p["p90_itl_ms"] > 0
                    and 1000 / p["p90_itl_ms"] >= target
                    and p.get("failed") in (None, 0)
                    and p[f"{token_type}_tps_per_gpu"] > 0]
        best = max(eligible, key=lambda p: p[f"{token_type}_tps_per_gpu"], default=None)
        row = dict(zip(SCOPE, key))
        row.update(status="eligible" if best else "no eligible observation",
                   point_id=best["id"] if best else None,
                   throughput_tps_per_gpu=best[f"{token_type}_tps_per_gpu"] if best else None,
                   usd_per_million_tokens=None, hourly_price=prices.get(row["hardware"]))
        if best and row["hourly_price"] is not None:
            row["usd_per_million_tokens"] = (
                row["hourly_price"] * 1e6
                / (3600 * row["throughput_tps_per_gpu"] * utilization)
            )
        results.append(row)
    return results


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_output(path):
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"output directory must be empty: {path}")
    return path


def finish(out, points, metadata, args):
    if not points:
        raise ValueError("no usable points; see raw.json and metadata.json")
    prices = json.loads(args.prices.read_text()) if args.prices else {}
    if not isinstance(prices, dict):
        raise ValueError("prices must be a JSON object: hardware -> USD/GPU-hour")
    summary = analyze(points, args.target, args.token_type, prices, args.utilization)
    bundle = dict(schema_version=1, metadata=metadata, points=points,
                  target=args.target, token_type=args.token_type,
                  prices=prices, utilization=args.utilization)
    write_json(out / "points.json", bundle)
    write_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    # Escape script delimiters even when metadata contains untrusted model names.
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (HERE / "report.html").read_text()
    (out / "report.html").write_text(template.replace("__BENCHMARK_DATA__", payload))
    print(f"{len(points)} points; {sum(r['status'] == 'eligible' for r in summary)} eligible series")
    print(f"Report: {out / 'report.html'}")


def fetch(args):
    out = prepare_output(args.output_dir)
    query = {"model": args.model, "date": args.date}
    url = API + "?" + urlencode(query)
    request = Request(url, headers={"User-Agent": "AIInfra-teaching-demo/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    (out / "raw.json").write_bytes(raw)
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("expected the InferenceX benchmark row array")
    points, skipped = [], Counter()
    for row in rows:
        try:
            point = public_point(row)
            if args.workload and point["workload"] != args.workload:
                continue
            points.append(point)
        except (KeyError, TypeError, ValueError) as exc:
            skipped[str(exc)] += 1
    metadata = dict(source_url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest(), as_of=args.date,
                    raw_rows=len(rows), selected_points=len(points), skipped=dict(skipped),
                    normalization="aggregate: max(P,D); disaggregated: P+D; total TPS / physical GPUs",
                    note="Published observations from different dates; consult each point's date and run_url.")
    write_json(out / "metadata.json", metadata)
    if not points:
        available = sorted({str(r.get("benchmark_type")) for r in rows})
        raise ValueError(f"no matching observations; benchmark types available: {available}")
    finish(out, points, metadata, args)


def commands(args, out):
    parsed = urlsplit(args.base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("--base-url must be an HTTP(S) origin without credentials or /v1")
    try:
        levels = [positive_int(c) for c in args.concurrency.split(",")]
    except argparse.ArgumentTypeError as exc:
        raise ValueError("concurrency must be comma-separated positive integers") from exc
    if len(set(levels)) != len(levels):
        raise ValueError("concurrency levels must be unique")
    if max(levels) > args.num_prompts:
        raise ValueError("--num-prompts must be >= every concurrency level")
    plans = []
    for c in levels:
        command = [args.client, "bench", "serve", "--backend", "openai",
                   "--base-url", args.base_url.rstrip("/"), "--endpoint", "/v1/completions",
                   "--model", args.model, "--dataset-name", "random",
                   "--random-input-len", str(args.isl), "--random-output-len", str(args.osl),
                   "--random-range-ratio", "0", "--num-prompts", str(args.num_prompts),
                   "--max-concurrency", str(c), "--request-rate", "inf", "--seed", "42",
                   "--num-warmups", "1", "--ignore-eos",
                   "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "50,90,99",
                   "--goodput", f"ttft:{args.ttft_slo_ms}", f"tpot:{args.tpot_slo_ms}",
                   "--save-result", "--result-dir", str(out),
                   "--result-filename", f"raw-conc-{c}.json"]
        if args.tokenizer:
            command.extend(["--tokenizer", args.tokenizer])
        plans.append((c, command))
    return plans


def bench(args):
    out = prepare_output(args.output_dir)
    plans = commands(args, out)
    metadata = dict(source="unofficial local run", created_at=datetime.now(timezone.utc).isoformat(),
                    model=args.model, hardware=args.hardware, framework=args.framework,
                    precision=args.precision, num_gpus=args.num_gpus, server_note=args.server_note,
                    workload=f"single_turn:{args.isl}/{args.osl}",
                    client_queue="closed-loop saturation; client waiting is outside ITL/TTFT",
                    request_slos_ms={"ttft": args.ttft_slo_ms, "tpot": args.tpot_slo_ms},
                    commands=[command for _, command in plans],
                    status="dry-run" if args.dry_run else "running")
    write_json(out / "metadata.json", metadata)
    if args.dry_run:
        for _, command in plans:
            print(shlex.join(command))
        return
    points = []
    try:
        result = subprocess.run([args.client, "--version"], check=True,
                                text=True, capture_output=True, timeout=30)
        version = result.stdout.strip()
        metadata["client_version"] = version
        for c, command in plans:
            print(f"Running concurrency={c}", flush=True)
            with (out / f"client-conc-{c}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout_seconds)
            raw_path = out / f"raw-conc-{c}.json"
            point = local_point(json.loads(raw_path.read_text()), args, c, version,
                                metadata["created_at"][:10])
            points.append(point)
            if point["failed"] or not point["completed"]:
                raise ValueError(f"concurrency {c} has failed requests; inspect raw result and log")
        metadata["status"] = "complete"
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        write_json(out / "metadata.json", metadata)
    finish(out, points, metadata, args)


def sample(args):
    out = prepare_output(args.output_dir)
    points = []
    for hardware, scale in (("SYNTHETIC-A", 1), ("SYNTHETIC-B", 1.25)):
        for c, itl, tps in ((1, 8, 60), (4, 12, 180), (16, 25, 360), (32, 45, 400)):
            points.append(dict(
                source="SYNTHETIC: UI demonstration only", id=f"{hardware}-{c}",
                model="SYNTHETIC-MODEL", workload="single_turn:1024/128",
                hardware=hardware, framework="simulation", precision="illustrative",
                spec_method="none", offload_mode="off", disagg=False,
                date="not measured", image="not measured", num_gpus=1, concurrency=c,
                output_tps_per_gpu=tps * scale, total_tps_per_gpu=tps * scale * 9,
                p90_itl_ms=itl / scale, p99_itl_ms=itl * 1.5 / scale,
                median_ttft_ms=50 * c, failed=0, completed=100, run_url=None))
    metadata = dict(source="Synthetic fixture; no hardware benchmark was performed")
    write_json(out / "metadata.json", metadata)
    finish(out, points, metadata, args)


def report(args):
    bundle = json.loads(args.input.read_text())
    if bundle.get("schema_version") != 1:
        raise ValueError("expected a schema_version=1 points.json bundle")
    points = bundle["points"]
    for p in points:
        for key in ("output_tps_per_gpu", "total_tps_per_gpu"):
            number(p[key], key, zero=True)
        for key in ("p90_itl_ms", "p99_itl_ms", "median_ttft_ms"):
            if p.get(key) is not None:
                number(p[key], key, zero=True)
    finish(prepare_output(args.output_dir), points, bundle["metadata"], args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("sample", sample), ("fetch", fetch), ("bench", bench), ("report", report)):
        p = sub.add_parser(name)
        p.set_defaults(function=function)
        p.add_argument("--output-dir", required=True, type=Path)
        p.add_argument("--target", type=float, default=50, help="minimum 1/P90 ITL, output tok/s")
        p.add_argument("--token-type", choices=("output", "total"), default="output")
        p.add_argument("--prices", type=Path, help="JSON: hardware -> USD/GPU-hour")
        p.add_argument("--utilization", type=float, default=1)
        if name == "report":
            p.add_argument("--input", required=True, type=Path, help="saved points.json bundle")
        if name == "fetch":
            p.add_argument("--model", default="DeepSeek-V4.1-Flash")
            p.add_argument("--date", required=True, help="as-of date YYYY-MM-DD")
            p.add_argument("--workload", help="exact workload, e.g. agentic_traces or single_turn:8192/1024")
        if name == "bench":
            p.add_argument("--base-url", required=True)
            p.add_argument("--model", required=True)
            p.add_argument("--hardware", required=True)
            p.add_argument("--num-gpus", required=True, type=positive_int)
            p.add_argument("--framework", choices=("vllm", "sglang", "other"), required=True)
            p.add_argument("--precision", required=True)
            p.add_argument("--server-note", required=True, help="server image/commit, TP/EP, cache config")
            p.add_argument("--spec-method", default="none")
            p.add_argument("--disagg", action="store_true")
            p.add_argument("--concurrency", default="1,4,8")
            p.add_argument("--num-prompts", default=32, type=positive_int)
            p.add_argument("--isl", default=1024, type=positive_int)
            p.add_argument("--osl", default=128, type=positive_int)
            p.add_argument("--ttft-slo-ms", default=1000, type=positive_int)
            p.add_argument("--tpot-slo-ms", default=50, type=positive_int)
            p.add_argument("--timeout-seconds", default=1800, type=positive_int)
            p.add_argument("--tokenizer")
            p.add_argument("--client", default="vllm")
            p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        number(args.target, "target")
        number(args.utilization, "utilization")
        if args.utilization > 1:
            raise ValueError("utilization must be <= 1")
        if args.command == "fetch":
            datetime.strptime(args.date, "%Y-%m-%d")
        if args.prices:
            prices = json.loads(args.prices.read_text())
            if not isinstance(prices, dict):
                raise ValueError("prices must be a JSON object")
            for hardware, price in prices.items():
                number(price, f"price for {hardware}")
        args.function(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
