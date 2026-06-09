# Agent Infra Reference

This directory documents the memory/skill/state/action infrastructure for research agent baselines.

## Recommended Baseline Ladder

1. Direct agent: use the current observation only.
2. ReAct agent: add explicit thought/action formatting and short trajectory history.
3. Memory agent: retrieve successful experiences and failure cases from previous episodes.
4. Skill agent: retrieve reusable task skills from a skill bank.
5. Memory + Skill agent: combine experience retrieval with skill retrieval.
6. RL-trained variants: reuse the same prompt builders and action parsers inside GRPO/RLOO/PPO rollout.

## Suggested Experiment Controls

- Keep model, tokenizer, decoding parameters, environment seed, max steps, and reward function fixed across baselines.
- Log action validity, success rate, reward, episode length, token cost, retrieved memory count, retrieved skill count, and failure category.
- Separate prompt-based baseline runs from RL fine-tuning runs.
- Store trajectories in JSONL so later memory and skill mining experiments can reuse the same data.
