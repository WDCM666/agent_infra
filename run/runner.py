"""
Core runner — wires together an LLM, an Agent, and an Environment.

The runner reads an :class:`ExperimentConfig`, creates the appropriate
components, and executes the agent–environment interaction loop.

Supports all agent types (react, reflexion, memory, skill, direct) × all
environments (alfworld, webshop, search, sokoban, gym_cards, appworld).

Usage::

    from run.config import load_yaml_config
    from run.runner import Runner

    cfg = load_yaml_config("run/configs/alfworld_react.yaml")
    runner = Runner(cfg)
    results = runner.run()
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from run.config import (
    EnvConfig,
    ExperimentConfig,
    _build_inner_env_name,
    _env_config_to_omega,
)
from run.llm_client import BaseLLMClient, get_llm_client

from agent_system.action import ActionParser
from agent_system.agents import (
    BaseAgent,
    DirectAgent,
    MemoryAgent,
    ReActAgent,
    ReflexionAgent,
    SkillAgent,
)
from agent_system.environments.base import EnvironmentManagerBase

logger = logging.getLogger(__name__)


# ======================================================================
# Runner
# ======================================================================

class Runner:
    """Top-level orchestrator for a single experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment specification (agent, env, LLM, run params).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm_client: BaseLLMClient = get_llm_client(config.llm)
        self.env_manager: EnvironmentManagerBase = self._create_env()
        self.agent: BaseAgent = self._create_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Run all episodes.  Returns a list of per-episode result dicts."""
        results = []
        for ep in range(self.config.num_episodes):
            if self.config.verbose:
                print(f"\n{'='*60}\nEpisode {ep + 1}/{self.config.num_episodes}\n{'='*60}")
            result = self._run_episode(ep)
            results.append(result)
            if self.config.verbose:
                self._print_episode_summary(result)
                self._print_timing(result)
        return results

    def close(self):
        """Release environment resources."""
        if self.env_manager is not None:
            self.env_manager.close()

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def _run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run a single episode and return trajectory + metrics."""
        if self.config.agent_type.lower() == "reflexion":
            return self._run_reflexion_episode(episode_idx)

        return self._run_single_attempt(episode_idx, trial_idx=0)

    def _run_reflexion_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run Reflexion-style retries on the same task after failed attempts."""
        num_trials = int(
            self.config.env.kwargs.get(
                "num_trials",
                self.config.env.kwargs.get("reflexion_trials", 3),
            )
        )
        num_trials = max(1, num_trials)
        if hasattr(self.agent, "reset_reflections"):
            self.agent.reset_reflections()

        trials: List[Dict[str, Any]] = []
        reflections: List[str] = []
        final_result: Optional[Dict[str, Any]] = None

        for trial_idx in range(num_trials):
            attempt = self._run_single_attempt(episode_idx, trial_idx=trial_idx)
            trials.append(attempt)
            final_result = attempt

            if attempt["success"]:
                break

            if trial_idx < num_trials - 1:
                reflection = self._generate_reflection(attempt["trajectory"], episode_idx, trial_idx)
                if reflection:
                    reflections.append(reflection)
                    if hasattr(self.agent, "add_reflection"):
                        self.agent.add_reflection(reflection)

        assert final_result is not None
        final_result = dict(final_result)
        final_result["num_trials"] = len(trials)
        final_result["reflections"] = reflections
        final_result["trials"] = trials
        final_result["elapsed_sec"] = sum(t["elapsed_sec"] for t in trials)
        final_result["num_steps"] = sum(t["num_steps"] for t in trials)
        final_result["total_reward"] = sum(t["total_reward"] for t in trials)
        final_result["timing"] = {
            key: round(sum((t.get("timing") or {}).get(key, 0) for t in trials), 3)
            for key in {"reset", "build_prompt", "llm_generate", "parse_action", "env_step"}
        }
        return final_result

    def _run_single_attempt(self, episode_idx: int, trial_idx: int = 0) -> Dict[str, Any]:
        """Run one environment attempt and return trajectory + metrics."""
        t0 = time.time()
        timing: Dict[str, float] = {"reset": 0, "build_prompt": 0, "llm_generate": 0, "parse_action": 0, "env_step": 0}

        _t = time.time()
        obs_dict, infos = self.env_manager.reset(kwargs={})
        timing["reset"] = time.time() - _t

        trajectory: List[Dict[str, Any]] = []
        success = False
        total_reward = 0.0

        for step in range(self.config.env.max_steps):
            _t = time.time()
            text_obs = self._extract_text_obs(obs_dict)
            prompt = self._build_prompt(text_obs, step, episode_idx)
            system_prompt = self._build_system_prompt()
            timing["build_prompt"] += time.time() - _t

            _t = time.time()
            model_output = self.llm_client.generate(prompt, system_prompt=system_prompt)
            timing["llm_generate"] += time.time() - _t

            _t = time.time()
            action = self.agent.parse_action(model_output)
            timing["parse_action"] += time.time() - _t

            _t = time.time()
            next_obs_dict, rewards, dones, infos = self.env_manager.step([action])
            timing["env_step"] += time.time() - _t

            reward = float(rewards[0]) if len(rewards) > 0 else 0.0
            done = bool(dones[0]) if len(dones) > 0 else False
            info = infos[0] if len(infos) > 0 else {}

            if self.config.verbose:
                print(f"--- Step {step + 1} ---")
                print(f"Action: {action}")
                print(f"Reward: {reward} | Done: {done}")

            trajectory.append({
                "step": step,
                "observation": text_obs,
                "model_output": model_output,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
            })

            total_reward += reward
            obs_dict = next_obs_dict

            if done:
                success = info.get("won", False)
                break

        elapsed = time.time() - t0
        return {
            "episode": episode_idx,
            "trial": trial_idx,
            "success": success,
            "total_reward": total_reward,
            "num_steps": len(trajectory),
            "elapsed_sec": elapsed,
            "trajectory": trajectory,
            "timing": {k: round(v, 3) for k, v in timing.items()},
        }

    def _generate_reflection(self, trajectory: List[Dict[str, Any]], episode_idx: int, trial_idx: int) -> str:
        """Ask the LLM for one concise Reflexion memory after a failed attempt."""
        lines = [
            "You are helping an ALFWorld agent improve after a failed attempt.",
            "Write one concise reflection that identifies the likely mistake and a better strategy for the next retry.",
            "Do not propose invalid commands. Keep it under 80 words.",
            f"Episode: {episode_idx}, failed trial: {trial_idx}",
            "",
            "Trajectory summary:",
        ]
        for step in trajectory[-20:]:
            info = step.get("info") or {}
            feedback = str(info.get("observation_text") or "")[:300].replace("\n", " ")
            lines.append(
                f"{step['step'] + 1}. action={step.get('action')!r}, "
                f"reward={step.get('reward')}, done={step.get('done')}, feedback={feedback!r}"
            )

        old_stop = getattr(self.llm_client.config, "stop", None)
        try:
            self.llm_client.config.stop = None
            reflection = self.llm_client.generate("\n".join(lines), system_prompt=None)
        finally:
            self.llm_client.config.stop = old_stop

        return " ".join(str(reflection).strip().split())[:800]

    # ------------------------------------------------------------------
    # Prompt building  (env's text obs + optional agent augmentation)
    # ------------------------------------------------------------------

    def _build_prompt(self, text_obs: str, step: int, episode_idx: int) -> str:
        """Return the full prompt sent to the LLM.

        Base prompt comes from the EnvironmentManager's ``build_text_obs``,
        which includes task description, history, and admissible actions.
        Agent-specific augmentations (reflections, memories, skills) are
        prepended when applicable.
        """
        if self.config.agent_type == "reflexion":
            return self._augment_reflexion(text_obs)
        elif self.config.agent_type == "memory":
            return self._augment_memory(text_obs)
        elif self.config.agent_type == "skill":
            return self._augment_skill(text_obs)
        else:
            # react / direct — use env prompt as-is
            return text_obs

    def _build_system_prompt(self) -> Optional[str]:
        if self.config.env.name.lower() != "alfworld":
            return None
        action_space = str(self.config.env.kwargs.get("action_space", "admissible")).lower()
        if action_space != "generation":
            return None
        prompt_style = str(self.config.env.kwargs.get("prompt_style", "grammar_react")).lower()
        if prompt_style == "official_react":
            return None
        if prompt_style == "official_grammar_react":
            from agent_system.environments.prompts.alfworld import ALFWORLD_OFFICIAL_GRAMMAR_SYSTEM_PROMPT

            return ALFWORLD_OFFICIAL_GRAMMAR_SYSTEM_PROMPT
        if prompt_style == "grammar_step_react":
            from agent_system.environments.prompts.alfworld import ALFWORLD_GRAMMAR_STEP_SYSTEM_PROMPT

            return ALFWORLD_GRAMMAR_STEP_SYSTEM_PROMPT
        if prompt_style == "official_grammar_sync_react":
            from agent_system.environments.prompts.alfworld import ALFWORLD_OFFICIAL_GRAMMAR_SYNC_SYSTEM_PROMPT

            return ALFWORLD_OFFICIAL_GRAMMAR_SYNC_SYSTEM_PROMPT
        if prompt_style == "direct_grammar":
            from agent_system.environments.prompts.alfworld import ALFWORLD_DIRECT_GRAMMAR_SYSTEM_PROMPT

            return ALFWORLD_DIRECT_GRAMMAR_SYSTEM_PROMPT

        from agent_system.environments.prompts.alfworld import ALFWORLD_GENERATION_SYSTEM_PROMPT

        return ALFWORLD_GENERATION_SYSTEM_PROMPT

    def _augment_reflexion(self, text_obs: str) -> str:
        if hasattr(self.agent, "reflections") and self.agent.reflections:
            reflection_text = "\n".join(
                f"- {r}" for r in self.agent.reflections[-3:]  # last 3
            )
            return f"Previous reflections:\n{reflection_text}\n\n{text_obs}"
        return text_obs

    def _augment_memory(self, text_obs: str) -> str:
        try:
            memories = self.agent.retrieve_memory(text_obs)
            if memories:
                mem_text = "\n".join(f"- {m}" for m in memories[:5])
                return f"Relevant past experiences:\n{mem_text}\n\n{text_obs}"
        except Exception:
            pass
        return text_obs

    def _augment_skill(self, text_obs: str) -> str:
        try:
            skills = self.agent.retrieve_skills(text_obs)
            if skills:
                skill_text = "\n".join(
                    f"- {s.name}: {s.description}" for s in skills[:5]
                )
                return f"Available skills:\n{skill_text}\n\n{text_obs}"
        except Exception:
            pass
        return text_obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_obs(obs_dict: Dict[str, Any]) -> str:
        """Extract the text observation string from the env's observation dict."""
        text = obs_dict.get("text")
        if text is None:
            return ""
        if isinstance(text, list):
            return text[0] if text else ""
        return str(text)

    def _print_episode_summary(self, result: Dict[str, Any]):
        outcome = "✅ SUCCESS" if result["success"] else "❌ FAILURE"
        print(
            f"\n{outcome} | steps={result['num_steps']} | "
            f"reward={result['total_reward']:.1f} | "
            f"time={result['elapsed_sec']:.1f}s"
        )

    def _print_timing(self, result: Dict[str, Any]):
        t = result.get("timing", {})
        if not t:
            return
        total = result["elapsed_sec"]
        steps = result["num_steps"]
        print(f"\n⏱  Timing breakdown ({steps} steps, {total:.1f}s total):")
        for label, sec in t.items():
            pct = sec / total * 100 if total > 0 else 0
            per_step = sec / steps * 1000 if steps > 0 else 0
            print(f"  {label:<20s}: {sec:7.1f}s ({pct:5.1f}%)  = {per_step:4.0f}ms/step")

    # ==================================================================
    # Agent factory
    # ==================================================================

    def _create_agent(self) -> BaseAgent:
        agent_type = self.config.agent_type.lower()
        action_parser = ActionParser()

        if agent_type == "react":
            return ReActAgent(prompt_builder=None, action_parser=action_parser)
        elif agent_type == "reflexion":
            return ReflexionAgent(prompt_builder=None, action_parser=action_parser)
        elif agent_type == "direct":
            return DirectAgent(prompt_builder=None, action_parser=action_parser)
        elif agent_type == "memory":
            return MemoryAgent(memory=None, retriever=None, prompt_builder=None, action_parser=action_parser)
        elif agent_type == "skill":
            return SkillAgent(skill_retriever=None, prompt_builder=None, action_parser=action_parser)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Supported: react, reflexion, memory, skill, direct")

    # ==================================================================
    # Environment factory
    # ==================================================================

    def _create_env(self) -> EnvironmentManagerBase:
        """Dispatch to the appropriate builder based on ``config.env.name``."""
        env_name = self.config.env.name.lower()
        if "alfworld" in env_name:
            return self._create_alfworld()
        elif "webshop" in env_name:
            return self._create_webshop()
        elif "search" in env_name:
            return self._create_search()
        elif "sokoban" in env_name:
            return self._create_sokoban()
        elif "gym_cards" in env_name or "gymcards" in env_name:
            return self._create_gym_cards()
        elif "appworld" in env_name:
            return self._create_appworld()
        else:
            raise ValueError(f"Unknown environment: {env_name}. Supported: alfworld, webshop, search, sokoban, gym_cards, appworld")

    # ------------------------------------------------------------------
    # Individual env builders
    # ------------------------------------------------------------------

    def _resolve_path(self, *parts: str) -> str:
        """Resolve a path relative to the ``run/`` directory."""
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, *parts)

    def _create_alfworld(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.alfworld import (
            alfworld_projection,
            build_alfworld_envs,
        )
        from agent_system.environments.env_manager import AlfWorldEnvironmentManager

        alf_config_path = self._resolve_path(
            "..", "agent_system", "environments", "env_package",
            "alfworld", "configs", "config_tw.yaml"
        )

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)
        env_kwargs = dict(self.config.env.kwargs)
        alf_config_path = env_kwargs.get("alf_config_path", alf_config_path)

        _envs = build_alfworld_envs(
            alf_config_path=alf_config_path,
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            resources_per_worker={"num_cpus": 1},
            is_train=False,
            env_kwargs=env_kwargs,
        )
        projection_f = partial(alfworld_projection)
        return AlfWorldEnvironmentManager(_envs, projection_f, omega_cfg)

    def _create_webshop(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs,
            webshop_projection,
        )
        from agent_system.environments.env_manager import WebshopEnvironmentManager

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)

        use_small = self.config.env.kwargs.get("use_small", True)
        if use_small:
            file_path = self._resolve_path(
                "..", "agent_system", "environments", "env_package",
                "webshop", "webshop", "data", "items_shuffle_1000.json"
            )
            attr_path = self._resolve_path(
                "..", "agent_system", "environments", "env_package",
                "webshop", "webshop", "data", "items_ins_v2_1000.json"
            )
        else:
            file_path = self._resolve_path(
                "..", "agent_system", "environments", "env_package",
                "webshop", "webshop", "data", "items_shuffle.json"
            )
            attr_path = self._resolve_path(
                "..", "agent_system", "environments", "env_package",
                "webshop", "webshop", "data", "items_ins_v2.json"
            )

        # Check data files exist (they need to be downloaded first)
        for fpath, fname in [(file_path, "items_shuffle"), (attr_path, "items_ins")]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"WebShop data file not found: {fpath}\n"
                    f"Please run the setup script to download the data:\n"
                    f"  cd agent_system/environments/env_package/webshop/webshop\n"
                    f"  bash setup.sh -d small"
                )

        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": self.config.env.kwargs.get("human_goals", True),
            "file_path": file_path,
            "attr_path": attr_path,
        }

        _envs = build_webshop_envs(
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            resources_per_worker={"num_cpus": 1},
            is_train=False,
            env_kwargs=env_kwargs,
        )
        projection_f = partial(webshop_projection)
        time.sleep(0.3)  # brief wait for env readiness (matching original code)
        return WebshopEnvironmentManager(_envs, projection_f, omega_cfg)

    def _create_search(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.search import (
            build_search_envs,
            search_projection,
        )
        from agent_system.environments.env_manager import SearchEnvironmentManager

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)

        _envs = build_search_envs(
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            is_train=False,
            env_config=omega_cfg.env,
        )
        projection_f = partial(search_projection)
        return SearchEnvironmentManager(_envs, projection_f, omega_cfg)

    def _create_sokoban(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.sokoban import (
            build_sokoban_envs,
            sokoban_projection,
        )
        from agent_system.environments.env_manager import SokobanEnvironmentManager

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)

        env_kwargs = {
            "dim_room": self.config.env.kwargs.get("dim_room", [7, 7]),
            "num_boxes": self.config.env.kwargs.get("num_boxes", 2),
            "max_steps": self.config.env.max_steps,
            "search_depth": self.config.env.kwargs.get("search_depth", 30),
        }

        _envs = build_sokoban_envs(
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            mode=self.config.env.kwargs.get("mode", "tiny_rgb_array"),
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker={"num_cpus": 1},
        )
        projection_f = partial(sokoban_projection)
        return SokobanEnvironmentManager(_envs, projection_f, omega_cfg)

    def _create_gym_cards(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.gym_cards import (
            build_gymcards_envs,
            gym_projection,
        )
        from agent_system.environments.env_manager import GymCardEnvironmentManager

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)
        env_name = omega_cfg.env.env_name

        _envs = build_gymcards_envs(
            env_name=env_name,
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            is_train=False,
            resources_per_worker={"num_cpus": 1},
        )
        projection_f = partial(gym_projection, env_name=env_name)
        return GymCardEnvironmentManager(_envs, projection_f, omega_cfg)

    def _create_appworld(self) -> EnvironmentManagerBase:
        from agent_system.environments.env_package.appworld import (
            appworld_projection,
            build_appworld_envs,
        )
        from agent_system.environments.env_manager import AppWorldEnvironmentManager

        inner_name = _build_inner_env_name(self.config.env)
        omega_cfg = _env_config_to_omega(self.config.env, inner_name)
        dataset_name = self.config.env.kwargs.get("dataset_name", "test_normal")

        _envs = build_appworld_envs(
            dataset_name=dataset_name,
            seed=self.config.env.seed,
            env_num=1,
            group_n=1,
            start_server_id=0,
            resources_per_worker={"num_cpus": 1},
        )
        projection_f = partial(appworld_projection)
        return AppWorldEnvironmentManager(_envs, projection_f, omega_cfg)
