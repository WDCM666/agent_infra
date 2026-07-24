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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config, path, value):
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


ALFWORLD_FLAT_OVERRIDE_PATHS = {
    "data_path": ("dataset", "data_path"),
    "eval_id_data_path": ("dataset", "eval_id_data_path"),
    "eval_ood_data_path": ("dataset", "eval_ood_data_path"),
    "num_train_games": ("dataset", "num_train_games"),
    "num_eval_games": ("dataset", "num_eval_games"),
    "env_type": ("env", "type"),
    "domain_randomization": ("env", "domain_randomization"),
    "task_types": ("env", "task_types"),
    "expert_timeout_steps": ("env", "expert_timeout_steps"),
    "expert_type": ("env", "expert_type"),
    "goal_desc_human_anns_prob": ("env", "goal_desc_human_anns_prob"),
    "hybrid_start_eps": ("env", "hybrid", "start_eps"),
    "hybrid_thor_prob": ("env", "hybrid", "thor_prob"),
    "hybrid_eval_mode": ("env", "hybrid", "eval_mode"),
    "thor_screen_width": ("env", "thor", "screen_width"),
    "thor_screen_height": ("env", "thor", "screen_height"),
    "thor_smooth_nav": ("env", "thor", "smooth_nav"),
    "thor_save_frames_to_disk": ("env", "thor", "save_frames_to_disk"),
    "thor_save_frames_path": ("env", "thor", "save_frames_path"),
    "controller_type": ("controller", "type"),
    "controller_debug": ("controller", "debug"),
    "load_receps": ("controller", "load_receps"),
    "mask_rcnn_pretrained_model_path": ("mask_rcnn", "pretrained_model_path"),
    "random_seed": ("general", "random_seed"),
    "use_cuda": ("general", "use_cuda"),
    "visdom": ("general", "visdom"),
    "task": ("general", "task"),
    "training_method": ("general", "training_method"),
    "save_path": ("general", "save_path"),
    "observation_pool_capacity": ("general", "observation_pool_capacity"),
    "hide_init_receptacles": ("general", "hide_init_receptacles"),
}


def apply_alfworld_overrides(config, env_kwargs):
    """
    Apply runner YAML overrides to the ALFWorld config.

    Supported forms:
    - env.kwargs.<flat_key>: common aliases listed in ALFWORLD_FLAT_OVERRIDE_PATHS
    - env.kwargs.<top_level_section>: nested overrides for dataset/env/controller/etc.
    - env.kwargs.config_overrides: final deep override with the same shape as config_tw.yaml
    """
    if not env_kwargs:
        return config

    top_level_sections = set(config.keys())
    reserved = {
        "eval_dataset",
        "action_space",
        "prompt_style",
        "alf_config_path",
        "config_overrides",
        "alf_config_overrides",
        "game_file_start",
        "game_file_count",
        "game_file_indices",
    }

    for key, value in env_kwargs.items():
        if key in reserved:
            continue
        if key in ALFWORLD_FLAT_OVERRIDE_PATHS:
            _set_nested(config, ALFWORLD_FLAT_OVERRIDE_PATHS[key], value)
        elif key in top_level_sections and isinstance(value, dict):
            _deep_update(config[key], value)

    explicit_overrides = env_kwargs.get("config_overrides") or env_kwargs.get("alf_config_overrides") or {}
    if explicit_overrides:
        _deep_update(config, explicit_overrides)

    return config


def apply_game_file_window(base_env, env_kwargs):
    """Restrict ALFWorld's collected game list to a shard.

    ``game_file_indices`` is used by the parallel runner to assign non-contiguous
    episode ids to each worker while still reusing one registered TextWorld env
    per worker process.
    """
    if not env_kwargs:
        return
    if (
        "game_file_indices" not in env_kwargs
        and "game_file_start" not in env_kwargs
        and "game_file_count" not in env_kwargs
    ):
        return

    game_files = getattr(base_env, "game_files", None)
    if game_files is None:
        return

    if "game_file_indices" in env_kwargs:
        indices = [int(i) for i in env_kwargs.get("game_file_indices") or []]
        base_env.game_files = [
            game_files[i] for i in indices
            if 0 <= i < len(game_files)
        ]
        base_env.num_games = len(base_env.game_files)
        print(f"Using ALFWorld game file indices count={base_env.num_games}, indices={indices}")
        return

    start = max(0, int(env_kwargs.get("game_file_start", 0) or 0))
    count = env_kwargs.get("game_file_count")
    end = None if count is None or int(count) < 0 else start + max(0, int(count))
    base_env.game_files = game_files[start:end]
    base_env.num_games = len(base_env.game_files)
    print(f"Using ALFWorld game file window start={start}, count={base_env.num_games}")

def get_obs_image(env):
    import torch
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action]
        try:
            obs, scores, dones, infos = self.env.step(actions)
            infos['observation_text'] = obs
        except Exception:
            # Environment error (e.g. invalid action) — return a no-op observation
            obs = ['Nothing happens.']
            scores = [0.0]
            dones = [False]
            infos = {'won': [False], 'admissible_commands': [[]], 'observation_text': obs}
            return obs, scores, dones, infos
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        config = apply_alfworld_overrides(config, env_kwargs)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        apply_game_file_window(base_env, env_kwargs)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
