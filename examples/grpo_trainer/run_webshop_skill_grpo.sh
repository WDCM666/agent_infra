#!/usr/bin/env bash
set -euo pipefail

python -m verl.trainer.main_ppo "$@" \
  env.env_name=webshop \
  +agent_infra.agent_type=skill \
  +agent_infra.skills.bank=agent_system/skills/banks/webshop_skills.json
