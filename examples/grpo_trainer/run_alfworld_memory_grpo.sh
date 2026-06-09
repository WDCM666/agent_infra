#!/usr/bin/env bash
set -euo pipefail

# Template: fill in model/data paths and set an agent memory config before running.
python -m verl.trainer.main_ppo "$@" \
  env.env_name=alfworld \
  +agent_infra.agent_type=memory \
  +agent_infra.memory.enabled=true
