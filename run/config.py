"""
Config dataclasses, YAML loading, and OmegaConf adapter for the standalone runner.

Usage:
    from run.config import load_config
    cfg = load_config("run/configs/alfworld_react.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM API configuration."""
    provider: str = "openai"          # "openai" | "anthropic"
    model: str = "gpt-4o"
    api_key: str = ""                 # ENV var name or literal key
    base_url: Optional[str] = None    # custom endpoint
    temperature: float = 0.0
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str = "alfworld"            # "alfworld" | "webshop" | "search" | "sokoban" | "gym_cards" | "appworld"
    history_length: int = 10
    max_steps: int = 50
    seed: int = 42
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "default"
    agent_type: str = "react"         # "react" | "reflexion" | "memory" | "skill" | "direct"
    llm: LLMConfig = field(default_factory=LLMConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    num_episodes: int = 1
    num_agents: int = 1
    verbose: bool = True


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    _resolve_env_vars(raw)

    llm_raw = raw.get("llm", {})
    env_raw = raw.get("env", {})

    llm = LLMConfig(
        provider=llm_raw.get("provider", "openai"),
        model=llm_raw.get("model", "gpt-4o"),
        api_key=llm_raw.get("api_key", ""),
        base_url=llm_raw.get("base_url"),
        temperature=llm_raw.get("temperature", 0.0),
        max_tokens=llm_raw.get("max_tokens", 512),
        stop=llm_raw.get("stop"),
        extra_body=llm_raw.get("extra_body", {}) or {},
    )

    env = EnvConfig(
        name=env_raw.get("name", "alfworld"),
        history_length=env_raw.get("history_length", 10),
        max_steps=env_raw.get("max_steps", 50),
        seed=env_raw.get("seed", 42),
        kwargs=env_raw.get("kwargs", {}),
    )

    return ExperimentConfig(
        name=raw.get("name", "default"),
        agent_type=raw.get("agent_type", "react"),
        llm=llm,
        env=env,
        num_episodes=raw.get("num_episodes", 1),
        num_agents=raw.get("num_agents", 1),
        verbose=raw.get("verbose", True),
    )


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` patterns in string values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _resolve_env_vars(v)
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.environ.get(env_var, obj)
    return obj


# ---------------------------------------------------------------------------
# OmegaConf adapter  (keeps existing EnvironmentManager subclasses happy)
# ---------------------------------------------------------------------------

def _env_config_to_omega(env_cfg: EnvConfig, env_name: str) -> OmegaConf:
    """Build a minimal OmegaConf that mimics the old veRL config structure."""
    base: Dict[str, Any] = {
        "env": {
            "env_name": env_name,
            "history_length": env_cfg.history_length,
            "max_steps": env_cfg.max_steps,
            "seed": env_cfg.seed,
            "rollout": {"n": 1},
            "resources_per_worker": {"num_cpus": 1},
        },
        "data": {
            "train_batch_size": 1,
            "val_batch_size": 1,
        },
    }

    # Env-specific sub-keys (keep the existing EnvironmentManager code paths)
    if "alfworld" in env_name.lower():
        base["env"]["alfworld"] = {
            "eval_dataset": env_cfg.kwargs.get("eval_dataset", "eval_in_distribution"),
            "action_space": env_cfg.kwargs.get("action_space", "admissible"),
            "prompt_style": env_cfg.kwargs.get("prompt_style", "grammar_react"),
        }
    elif "webshop" in env_name.lower():
        base["env"]["webshop"] = {
            "use_small": env_cfg.kwargs.get("use_small", True),
            "human_goals": env_cfg.kwargs.get("human_goals", True),
        }
    elif "sokoban" in env_name.lower():
        base["env"]["sokoban"] = {
            "dim_room": env_cfg.kwargs.get("dim_room", [7, 7]),
            "num_boxes": env_cfg.kwargs.get("num_boxes", 2),
            "search_depth": env_cfg.kwargs.get("search_depth", 30),
            "mode": env_cfg.kwargs.get("mode", "tiny_rgb_array"),
        }
    elif "gym_cards" in env_name.lower():
        base["env"]["env_name"] = env_cfg.kwargs.get("env_name", env_name)
        base["env"]["env_name"] = env_cfg.kwargs.get("env_name", env_name)
    elif "search" in env_name.lower():
        pass  # search env doesn't need extra fields
    elif "appworld" in env_name.lower():
        pass  # appworld env doesn't need extra fields

    return OmegaConf.create(base)


def _build_inner_env_name(env_cfg: EnvConfig) -> str:
    """Map the short env name to the internal env_name string."""
    name = env_cfg.name.lower()
    mapping = {
        "alfworld": "alfworld/AlfredTWEnv",
        "webshop": "webshop",
        "search": "search",
        "sokoban": "sokoban",
        "gym_cards": env_cfg.kwargs.get("env_name", "gym_cards/numberline"),
        "appworld": "appworld",
    }
    return mapping.get(name, name)
