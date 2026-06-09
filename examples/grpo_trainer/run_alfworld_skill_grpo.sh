#!/usr/bin/env bash
set -euo pipefail

python -m verl.trainer.main_ppo "$@" \
  env.env_name=alfworld \
  +agent_infra.agent_type=skill \
  +agent_infra.skills.bank=agent_system/skills/banks/alfworld_skills.json
