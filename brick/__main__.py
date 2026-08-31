"""
CLI entry point for AIOS inference engine.

Usage:
    python -m aios --model /path/to/Qwen3-0.6B --prompt "Who are you?"
    python -m aios --model /path/to/Qwen3-0.6B --prompt "Hi" --prompt "Hello" --max-running-reqs 2
"""

import argparse
import time

from aios import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="AIOS LLM Inference Engine")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory or HuggingFace model name")
    parser.add_argument("--prompt", type=str, action="append", default=None,
                        help="Input prompt (repeat the flag for a batch)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (-1 = disabled)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g. cuda, cuda:0, cuda:1)")
    parser.add_argument("--max-running-reqs", type=int, default=None,
                        help="Cap concurrently running reqs (defaults to batch size)")
    args = parser.parse_args()

    prompts = args.prompt or ["Who are you?"]

    print(f"Loading model from {args.model}...")
    t0 = time.perf_counter()
    llm = LLM(args.model, device=args.device)
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.1f}s\n")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    results = llm.generate(prompts, sampling_params, max_running_reqs=args.max_running_reqs)

    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"=== prompt {i}: {prompts[i]!r} ===")
        print(r["text"])


if __name__ == "__main__":
    main()
