import json
import os


# --------------------- ALFWorld --------------------- #
ALFWORLD_ADMISSIBLE_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one admissible action copied verbatim from the list.
The Action line is mandatory. Do not write any extra lines.
"""

ALFWORLD_ADMISSIBLE_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one admissible action copied verbatim from the list.
The Action line is mandatory. Do not write any extra lines.
"""

ALFWORLD_GRAMMAR_INSTRUCTIONS = """
You are an expert agent operating in the ALFRED Embodied Environment.
This is action-grammar-constrained text generation. You are not given the current admissible action list.
Use only one command matching one of these action grammars:
- go to <receptacle>
- open <receptacle>
- close <receptacle>
- take <object> from <receptacle>
- put <object> in <receptacle>
- put <object> on <receptacle>
- clean <object> with <receptacle>
- heat <object> with <receptacle>
- cool <object> with <receptacle>
- use <toggleable object>

For official ReAct examples, "put <object> in/on <receptacle>" is also accepted as
the official put-action surface form.
""".strip()

ALFWORLD_GENERATION_SYSTEM_PROMPT = f"""
{ALFWORLD_GRAMMAR_INSTRUCTIONS}

For every turn, respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one command matching the grammar.
The Action line is mandatory. Do not write any extra lines.
""".strip()

ALFWORLD_OFFICIAL_GRAMMAR_SYSTEM_PROMPT = f"""
{ALFWORLD_GRAMMAR_INSTRUCTIONS}

Follow the ReAct trajectory format shown in the user prompt.
For every turn, output exactly one line:
- either a thought line starting with "think:"
- or one executable action matching the grammar above
Do not write "Thought:" or "Action:" labels in this mode.
""".strip()

ALFWORLD_GRAMMAR_STEP_SYSTEM_PROMPT = f"""
{ALFWORLD_GRAMMAR_INSTRUCTIONS}

For every turn, output exactly one line:
- either a thought line starting with "think:"
- or one executable action matching the grammar above
Do not write "Thought:" or "Action:" labels in this mode.
""".strip()

ALFWORLD_OFFICIAL_GRAMMAR_SYNC_SYSTEM_PROMPT = f"""
{ALFWORLD_GRAMMAR_INSTRUCTIONS}

Use the official ReAct examples in the user prompt as task-solving demonstrations.
For every turn, respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one executable action matching the grammar above.
Do not output "think:" in this mode.
The Action line is mandatory. Do not write any extra lines.
""".strip()

ALFWORLD_DIRECT_GRAMMAR_SYSTEM_PROMPT = f"""
{ALFWORLD_GRAMMAR_INSTRUCTIONS}

You are a direct-action baseline. Do not reason out loud.
For every turn, output exactly one executable action matching the grammar above.
Do not output "think:", "Thought:", "Action:", XML tags, explanations, or extra lines.
""".strip()

ALFWORLD_DIRECT_GENERATION_TEMPLATE = """
You are operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Output exactly one executable action.
"""

ALFWORLD_GENERATION_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action.
Respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one command matching the grammar.
The Action line is mandatory. Do not write any extra lines.
"""

ALFWORLD_GENERATION_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
Respond with exactly two lines:
Thought: at most 12 words about the immediate next step.
Action: one command matching the grammar.
The Action line is mandatory. Do not write any extra lines.
"""

ALFWORLD_GRAMMAR_STEP_HEADER = "Interact with a household to solve a task.\nHere is the task.\n"


def build_alfworld_grammar_step_prompt(interaction_text):
    """Build a no-few-shot ReAct prompt that uses official one-line turns."""
    return ALFWORLD_GRAMMAR_STEP_HEADER + interaction_text

# Backwards-compatible aliases for older imports.
ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_ADMISSIBLE_TEMPLATE_NO_HIS
ALFWORLD_TEMPLATE = ALFWORLD_ADMISSIBLE_TEMPLATE


ALFWORLD_OFFICIAL_REACT_TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

ALFWORLD_OFFICIAL_REACT_HEADER = "Interact with a household to solve a task. Here are two examples.\n"

_ALFWORLD_OFFICIAL_REACT_EXAMPLES = None


def _load_alfworld_official_react_examples():
    """Load the official ALFWorld ReAct few-shot trajectories from ysymyth/ReAct."""
    global _ALFWORLD_OFFICIAL_REACT_EXAMPLES
    if _ALFWORLD_OFFICIAL_REACT_EXAMPLES is None:
        path = os.path.join(os.path.dirname(__file__), "alfworld_official_react.json")
        with open(path, "r") as f:
            _ALFWORLD_OFFICIAL_REACT_EXAMPLES = json.load(f)
    return _ALFWORLD_OFFICIAL_REACT_EXAMPLES


def _official_react_prefix_from_gamefile(gamefile):
    if not gamefile:
        return None
    task_name = os.path.basename(os.path.dirname(os.path.dirname(str(gamefile))))
    for task_prefix, prompt_prefix in ALFWORLD_OFFICIAL_REACT_TASK_PREFIXES.items():
        if task_name.startswith(task_prefix):
            return prompt_prefix
    return None


def _official_react_prefix_from_task(task_description):
    task = str(task_description or "").lower()
    if "two" in task:
        return "puttwo"
    if "clean" in task:
        return "clean"
    if "heat" in task or "hot" in task:
        return "heat"
    if "cool" in task:
        return "cool"
    if "look at" in task or "examine" in task or "desklamp" in task:
        return "examine"
    return "put"


def build_alfworld_official_react_prompt(gamefile, task_description, interaction_text):
    """Build the official ReAct prompt layout for an ALFWorld episode."""
    examples = _load_alfworld_official_react_examples()
    prefix = (
        _official_react_prefix_from_gamefile(gamefile)
        or _official_react_prefix_from_task(task_description)
    )
    return (
        ALFWORLD_OFFICIAL_REACT_HEADER
        + examples[f"react_{prefix}_1"]
        + examples[f"react_{prefix}_0"]
        + "\nHere is the task.\n"
        + interaction_text
    )
