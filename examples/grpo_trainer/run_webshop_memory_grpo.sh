#!/usr/bin/env bash
set -euo pipefail

python -m verl.trainer.main_ppo "$@" \
  env.env_name=webshop \
  +agent_infra.agent_type=memory \
  +agent_infra.memory.enabled=true
