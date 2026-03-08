#!/usr/bin/env python3
"""Run the tool-calling benchmark against an OpenAI-compatible server."""

import argparse
import json
import os
import sys
import time

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
    hard_total = len(HARD_PROMPT_INDICES)

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
                        pass  # wrong tool, no credit
                    else:
                        action_correct += 0.5  # partial
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
        "action": action_score,
        "restraint": restraint_score,
        "agent": agent_score,
        "hard_correct": hard_correct,
        "hard_total": hard_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8095/v1")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-8bit")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--name", default=None, help="Display name")
    args = parser.parse_args()

    name = args.name or args.model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Tool-Calling Benchmark: {name}")
    print(f"Server: {args.base_url}")
    print(f"Runs: {args.num_runs}")
    print(f"{'='*60}\n")

    all_scores = []
    all_latencies = []
    
    for run in range(args.num_runs):
        print(f"--- Run {run+1}/{args.num_runs} ---")
        results = []
        for i, prompt in enumerate(TEST_PROMPTS):
            r = run_one_openai(args.base_url, args.model, prompt)
            tag = r["tool_name"] or ("(none)" if not r["error"] else "ERROR")
            ms = r["latency_ms"]
            all_latencies.append(ms)
            status = "✅" if (
                (i in RESTRAINT_INDICES and not r["tool_called"]) or
                (i in TOOL_CALL_INDICES and r["tool_called"] and r["valid_args"])
            ) else "❌"
            print(f"  P{i+1:>2}: {status} {tag:<20s} {ms:>5d}ms")
            results.append(r)
        
        scores = score_run(results)
        all_scores.append(scores)
        print(f"  → Agent: {scores['agent']:.3f} (Action: {scores['action']:.3f}, Restraint: {scores['restraint']:.3f}, Hard: {scores['hard_correct']}/{scores['hard_total']})")
        print()

    # Aggregate
    avg_agent = sum(s["agent"] for s in all_scores) / len(all_scores)
    avg_action = sum(s["action"] for s in all_scores) / len(all_scores)
    avg_restraint = sum(s["restraint"] for s in all_scores) / len(all_scores)
    avg_hard = sum(s["hard_correct"] for s in all_scores) / len(all_scores)
    avg_latency = sum(all_latencies) / len(all_latencies)
    p50 = sorted(all_latencies)[len(all_latencies)//2]

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {name} ({args.num_runs} runs)")
    print(f"{'='*60}")
    print(f"  Agent Score:     {avg_agent:.3f}")
    print(f"  Action Score:    {avg_action:.3f}")
    print(f"  Restraint Score: {avg_restraint:.3f}")
    print(f"  Hard Prompts:    {avg_hard:.1f}/3 avg")
    print(f"  Avg Latency:     {avg_latency:.0f}ms")
    print(f"  P50 Latency:     {p50}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
