#!/usr/bin/env python
"""
CLI entry point for the standalone agent-environment runner.

Usage::

    python -m run.run --config run/configs/alfworld_react.yaml
    python -m run.run --config run/configs/webshop_react.yaml --episodes 5

Or directly::

    python run/run.py --config run/configs/alfworld_react.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

# Make sure the project root is on sys.path so ``agent_system`` is importable.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from run.config import ExperimentConfig, load_yaml_config


# ---------------------------------------------------------------------------
# JSON helper — convert numpy types to native Python
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types in *obj* to JSON-serializable values."""
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if np is not None and isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _episode_assignments(num_episodes: int, num_agents: int) -> List[List[int]]:
    """Return round-robin episode id assignments for parallel workers.

    The previous contiguous sharding could strand a worker with several long
    ALFWorld episodes while other workers sat idle.  Round-robin keeps each
    worker's environment reusable, but usually balances heterogeneous task
    lengths better.
    """
    if num_episodes <= 0:
        return []
    worker_count = max(1, min(num_agents, num_episodes))
    assignments: List[List[int]] = [[] for _ in range(worker_count)]
    for episode_idx in range(num_episodes):
        assignments[episode_idx % worker_count].append(episode_idx)
    return assignments


def _run_worker(cfg: ExperimentConfig, worker_id: int, episode_indices: List[int]) -> List[dict]:
    """Run a shard of episode ids in a separate process."""
    from run.runner import Runner

    if not episode_indices:
        return []

    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.num_agents = 1
    worker_cfg.num_episodes = len(episode_indices)
    worker_cfg.verbose = False
    worker_cfg.env.seed = cfg.env.seed + episode_indices[0] * 9973
    if cfg.env.name.lower() == "alfworld":
        worker_cfg.env.kwargs = copy.deepcopy(cfg.env.kwargs)
        trial_count = 1
        if cfg.agent_type.lower() == "reflexion":
            trial_count = int(
                worker_cfg.env.kwargs.get(
                    "num_trials",
                    worker_cfg.env.kwargs.get("reflexion_trials", 3),
                )
            )
            trial_count = max(1, trial_count)
        worker_cfg.env.kwargs["game_file_indices"] = [
            episode_idx
            for episode_idx in episode_indices
            for _ in range(trial_count)
        ]

    runner = Runner(worker_cfg)
    try:
        results = runner.run()
    finally:
        runner.close()

    for local_idx, result in enumerate(results):
        result["episode"] = episode_indices[local_idx]
        result["worker_id"] = worker_id
    return results


def _run_experiment(cfg: ExperimentConfig) -> List[dict]:
    from run.runner import Runner

    num_agents = max(1, int(cfg.num_agents))
    if num_agents == 1 or cfg.num_episodes <= 1:
        runner = Runner(cfg)
        try:
            return runner.run()
        finally:
            runner.close()

    assignments = _episode_assignments(cfg.num_episodes, num_agents)
    print(f"Parallel workers: {len(assignments)}")
    for worker_id, episode_indices in enumerate(assignments):
        print(f"  worker {worker_id}: episodes {episode_indices}")

    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(_run_worker, cfg, worker_id, episode_indices)
            for worker_id, episode_indices in enumerate(assignments)
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    results.sort(key=lambda item: item["episode"])
    return results


def main(cfg: ExperimentConfig, config_path: str = "", output_dir: Optional[str] = None):
    print(f"Experiment: {cfg.name}")
    print(f"Agent:      {cfg.agent_type}")
    print(f"Env:        {cfg.env.name}")
    print(f"LLM:        {cfg.llm.provider}/{cfg.llm.model}")
    print(f"Episodes:   {cfg.num_episodes}")
    print(f"Agents:     {cfg.num_agents}")
    print(f"Max steps:  {cfg.env.max_steps}")
    print()

    # ---- run ------------------------------------------------------------------
    run_t0 = time.time()
    results = _run_experiment(cfg)
    wall_time = time.time() - run_t0

    # ---- summary --------------------------------------------------------------
    successes = sum(1 for r in results if r["success"])
    total_steps = sum(r["num_steps"] for r in results)
    avg_steps = total_steps / len(results) if results else 0
    total_time = sum(r["elapsed_sec"] for r in results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {successes}/{len(results)} successful")
    print(f"Avg steps: {avg_steps:.1f}  |  Episode time sum: {total_time:.1f}s  |  Wall time: {wall_time:.1f}s")
    print(f"{'='*60}")

    # ---- save -----------------------------------------------------------------
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{cfg.name}_{timestamp}.json")

    summary_data = _to_json_safe({
        "config": config_path,
        "name": cfg.name,
        "agent_type": cfg.agent_type,
        "env_name": cfg.env.name,
        "env_kwargs": cfg.env.kwargs,
        "action_space": cfg.env.kwargs.get("action_space"),
        "prompt_style": cfg.env.kwargs.get("prompt_style"),
        "llm_model": cfg.llm.model,
        "num_episodes": len(results),
        "num_agents": cfg.num_agents,
        "successes": successes,
        "success_rate": successes / len(results) if results else 0,
        "avg_steps": avg_steps,
        "total_time_sec": total_time,
        "wall_time_sec": wall_time,
        "timestamp": timestamp,
        "results": [
            {
                "episode": r["episode"],
                "worker_id": r.get("worker_id", 0),
                "success": r["success"],
                "total_reward": r["total_reward"],
                "num_steps": r["num_steps"],
                "elapsed_sec": r["elapsed_sec"],
                "avg_sec_per_step": r["elapsed_sec"] / r["num_steps"] if r["num_steps"] else 0,
                "num_trials": r.get("num_trials", 1),
                "reflections": r.get("reflections", []),
                "timing": r.get("timing", {}),
            }
            for r in results
        ],
        "trajectories": [r["trajectory"] for r in results],
        "trials": [r.get("trials") for r in results],
    })

    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agent in a benchmark environment")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML experiment config file",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=None,
        help="Override num_episodes from config",
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=None,
        help="Number of parallel agent worker processes",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for result output (default: run/outputs/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=None,
        help="Print per-step info",
    )
    args = parser.parse_args()

    # Load config and apply CLI overrides (before passing to main)
    cfg = load_yaml_config(args.config)
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.agents is not None:
        cfg.num_agents = args.agents
    if args.verbose is not None:
        cfg.verbose = args.verbose

    main(cfg, config_path=args.config, output_dir=args.output_dir)
