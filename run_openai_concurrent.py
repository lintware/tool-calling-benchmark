#!/usr/bin/env python3
"""Run the tool-calling benchmark with concurrent runs against an OpenAI-compatible server."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bench_config import (
    TEST_PROMPTS, TOOLS, RESTRAINT_INDICES, TOOL_CALL_INDICES,
    EXPECTED_TOOLS, WRONG_TOOL_MAP, HARD_PROMPT_INDICES, P8_REQUIRED_TOOLS,
)
from lib.openai_backend import run_one_openai


def score_run(results):
    """Score a single run (list of 12 prompt results)."""
    action_correct = 0
    action_total = len(TOOL_CALL_INDICES)
    restraint_correct = 0
    restraint_total = len(RESTRAINT_INDICES)
    hard_correct = 0

    for i, r in enumerate(results):
        if i in RESTRAINT_INDICES:
            if not r["tool_called"]:
                restraint_correct += 1
        elif i in TOOL_CALL_INDICES:
            if r["tool_called"] and r["valid_args"]:
                if i in EXPECTED_TOOLS:
                    if r["tool_name"] == EXPECTED_TOOLS[i]:
                        action_correct += 1
                        if i in HARD_PROMPT_INDICES:
                            hard_correct += 1
                    elif i in WRONG_TOOL_MAP and r["tool_name"] in WRONG_TOOL_MAP[i]:
                        pass
                    else:
                        action_correct += 0.5
                elif i == 7:  # P8: multi-tool
                    called_tools = {tc["name"] for tc in r["all_tool_calls"] if tc.get("valid", True)}
                    if P8_REQUIRED_TOOLS.issubset(called_tools):
                        action_correct += 1
                    elif called_tools & P8_REQUIRED_TOOLS:
                        action_correct += 0.5
                else:
                    action_correct += 1

    action_score = action_correct / action_total if action_total else 0
    restraint_score = restraint_correct / restraint_total if restraint_total else 0
    agent_score = (action_score + restraint_score) / 2

    return {
        "action": round(action_score, 3),
        "restraint": round(restraint_score, 3),
        "agent": round(agent_score, 3),
        "hard_correct": hard_correct,
        "hard_total": len(HARD_PROMPT_INDICES),
    }


def run_single_run(run_idx, base_url, model):
    """Execute one complete run (12 prompts sequentially) and return results."""
    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        r = run_one_openai(base_url, model, prompt)
        results.append(r)
    scores = score_run(results)
    return run_idx, results, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8095/v1")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-8bit")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default=None, help="Save JSON results to file")
    args = parser.parse_args()

    name = args.name or args.model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Tool-Calling Benchmark (Concurrent): {name}")
    print(f"Server: {args.base_url}")
    print(f"Runs: {args.num_runs}, Concurrency: {args.concurrency}")
    print(f"{'='*60}\n")

    all_scores = []
    all_results = []
    all_latencies = []

    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_single_run, i, args.base_url, args.model): i
            for i in range(args.num_runs)
        }
        for future in as_completed(futures):
            run_idx, results, scores = future.result()
            all_scores.append(scores)
            all_results.append({"run": run_idx, "results": results, "scores": scores})
            for r in results:
                all_latencies.append(r["latency_ms"])
            print(f"  Run {run_idx+1:>2}: Agent={scores['agent']:.3f} Action={scores['action']:.3f} Restraint={scores['restraint']:.3f} Hard={scores['hard_correct']}/{scores['hard_total']}")

    wall_time = time.time() - t_wall_start

    # Aggregate
    avg_agent = sum(s["agent"] for s in all_scores) / len(all_scores)
    avg_action = sum(s["action"] for s in all_scores) / len(all_scores)
    avg_restraint = sum(s["restraint"] for s in all_scores) / len(all_scores)
    avg_hard = sum(s["hard_correct"] for s in all_scores) / len(all_scores)
    avg_latency = sum(all_latencies) / len(all_latencies)
    p50 = sorted(all_latencies)[len(all_latencies) // 2]

    summary = {
        "model": args.model,
        "name": name,
        "num_runs": args.num_runs,
        "concurrency": args.concurrency,
        "agent_score": round(avg_agent, 3),
        "action_score": round(avg_action, 3),
        "restraint_score": round(avg_restraint, 3),
        "hard_prompts_avg": round(avg_hard, 1),
        "avg_latency_ms": round(avg_latency),
        "p50_latency_ms": p50,
        "wall_time_s": round(wall_time, 1),
        "total_requests": len(all_latencies),
        "hardware": "Apple M3 Ultra, 192GB",
        "backend": "mlx_lm.server" if "mlx" in args.model.lower() else "llama.cpp",
    }

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {name} ({args.num_runs} runs, {args.concurrency} concurrent)")
    print(f"{'='*60}")
    print(f"  Agent Score:     {avg_agent:.3f}")
    print(f"  Action Score:    {avg_action:.3f}")
    print(f"  Restraint Score: {avg_restraint:.3f}")
    print(f"  Hard Prompts:    {avg_hard:.1f}/3 avg")
    print(f"  Avg Latency:     {avg_latency:.0f}ms")
    print(f"  P50 Latency:     {p50}ms")
    print(f"  Wall Time:       {wall_time:.1f}s")
    print(f"  Total Requests:  {len(all_latencies)}")
    print(f"{'='*60}")

    if args.output:
        out = {"summary": summary, "runs": sorted(all_results, key=lambda x: x["run"])}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
