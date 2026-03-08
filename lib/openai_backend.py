"""OpenAI-compatible backend for local servers (MLX, llama.cpp, etc.)."""

import json
import time
import requests

from lib.bench_config import TOOLS

def run_one_openai(base_url: str, model: str, prompt: str) -> dict:
    """Run a single prompt against an OpenAI-compatible server with native tool calling."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "tool_called": False, "tool_name": None, "valid_args": None,
            "latency_ms": round(elapsed), "error": str(e), "raw_content": None,
            "all_tool_calls": [],
        }
    elapsed = (time.perf_counter() - t0) * 1000

    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []

    if not tool_calls:
        return {
            "tool_called": False, "tool_name": None, "valid_args": None,
            "latency_ms": round(elapsed), "error": None,
            "raw_content": msg.get("content"),
            "all_tool_calls": [],
        }

    all_tc = []
    for tc in tool_calls:
        fname = tc["function"]["name"]
        raw_args = tc["function"].get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            valid = True
        except (json.JSONDecodeError, TypeError):
            args = raw_args
            valid = False
        all_tc.append({"name": fname, "arguments": args, "valid": valid})

    first = all_tc[0]
    return {
        "tool_called": True,
        "tool_name": first["name"],
        "valid_args": first["valid"],
        "latency_ms": round(elapsed),
        "error": None,
        "raw_content": msg.get("content"),
        "all_tool_calls": all_tc,
    }
