# Memory / Skill Agent Baselines

This directory is for prompt-based baselines before wiring the same components into RL training.

Suggested first baselines:

- Direct: observation -> action.
- ReAct: observation + short history -> thought/action.
- Memory: observation + retrieved experiences/failures -> action.
- Skill: observation + retrieved skill snippets -> action.
- Memory + Skill: combine retrieved experience and skill prompts.
