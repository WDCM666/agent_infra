# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple
import re


def _strip_action_text(text: str) -> str:
    """Extract a likely executable command from raw model text or parsed output."""
    text = str(text).strip()
    if not text:
        return ""

    match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE)
        if match:
            text = match.group(1)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = lines[-1] if lines else text

    text = text.strip()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = text.strip("`'\" \t\r\n")
    text = text.rstrip(".。;；,，")
    return " ".join(text.lower().split())


def _normalize_pool(action_pool: Optional[List[str]]) -> Tuple[dict, List[Tuple[str, str]]]:
    if not action_pool:
        return {}, []

    normalized = []
    exact = {}
    for action in action_pool:
        norm = " ".join(str(action).strip().lower().split())
        if not norm or norm == "help":
            continue
        normalized.append((norm, action))
        exact[norm] = action
    return exact, normalized


def _put_equivalent_forms(candidate: str) -> List[str]:
    match = re.fullmatch(r"put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)", candidate)
    if not match:
        return []

    obj = match.group(1).strip()
    receptacle = match.group(2).strip()
    return [
        f"put {obj} in/on {receptacle}",
        f"put {obj} in {receptacle}",
        f"put {obj} on {receptacle}",
        f"move {obj} to {receptacle}",
    ]


def _match_admissible(candidate: str, action_pool: Optional[List[str]]) -> Tuple[str, int]:
    exact, normalized_pool = _normalize_pool(action_pool)
    if not candidate:
        return candidate, 0

    if candidate in exact:
        return exact[candidate], 1

    for variant in _put_equivalent_forms(candidate):
        if variant in exact:
            return exact[variant], 1

    # Accept one unambiguous admissible command embedded in a noisy line.
    matches = [original for norm, original in normalized_pool if re.search(rf"\b{re.escape(norm)}\b", candidate)]
    if len(matches) == 1:
        return matches[0], 1

    return candidate, 0


def alfworld_projection(actions: List[str], action_pools: List[List[str]], action_space: str = "admissible"):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    action_space = str(action_space or "admissible").lower()
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = str(actions[i])
        candidate = _strip_action_text(original_str)
        matched_action, is_valid = _match_admissible(candidate, action_pools[i])

        if action_space == "generation":
            actions[i] = matched_action if is_valid else candidate
        else:
            actions[i] = matched_action

        valids[i] = is_valid

        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
