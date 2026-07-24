# Installation
## Install veRL
```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install vllm==0.11.0

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

## Install Supported Environments
> ⚠️ **Important:** 
To run an agent in any of these environments, you must first install and configure the corresponding environment. We strongly recommend installing ***each environment in its own dedicated conda environment*** to avoid potential package version conflicts.

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

### 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---

### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

### 4. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
### 5. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 6. AppWorld (Experimental)
Install AppWorld package
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
```
You can ignore the warning of incompatibility for appworld, because we don't run appworld in `verl-agent` environment.

Create a dedicated conda environment `appworld` for the AppWorld server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```


<!-- > ⚠️ **Important:**  
To run an agent in any of these environments, you must first install and configure the corresponding environment. Please refer to the [Environment Setup Guide](agent_system/environments/README.md) for step-by-step installation instructions. -->

# Run Examples
## RL Training
We provide out-of-the-box scripts in the ["examples/"](./examples/) directory for training agents in different environments.

Here are some examples:
### 1. GiGPO
GiGPO is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups gather repeated states across trajectories to compute relative advantages for individual actions.

GiGPO is fully critic-free, maintains the same GPU memory footprint and LLM rollout cost as GRPO, yet achieves significantly better training efficiency and performance.

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_search.sh # Search
```
```bash
bash examples/gigpo_trainer/run_sokoban.sh # Sokoban
```
### 2. GRPO
GRPO is a critic-free algorithm that estimates relative advantages based on a group of full episode trajectories.
```bash
bash examples/grpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/grpo_trainer/run_webshop.sh # WebShop
```
### 3. PPO
PPO is a classic actor-critic algorithm that updates the policy using a clipped objective to ensure stable learning. It requires a separate value network (critic) to estimate state values.
```bash
bash examples/ppo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/ppo_trainer/run_webshop.sh # WebShop
```
### 4. RLOO
For RLOO, we use a leave-one-out estimate and the PPO-clip update (instead of the REINFORCE update), making it closer to [LOOP](https://arxiv.org/abs/2502.01600).
```bash
bash examples/rloo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/rloo_trainer/run_webshop.sh # WebShop
```
### 5. DAPO
DAPO enhances GRPO with techniques like dynamic sampling and clip-higher.
```bash
bash examples/dapo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/dapo_trainer/run_webshop.sh # WebShop
```
### 6. GiGPO (dynamic)
GiGPO uses dynamic sampling and clip-higher from DAPO
```bash
bash examples/gigpo_dynamic_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_dynamic_trainer/run_webshop.sh # WebShop
```

## LoRA
```bash
bash examples/gigpo_trainer/run_alfworld_lora.sh
```

## Prompt-based Agent with GPT-4o
We also provide a prompt-based GPT-4o agent.
```bash
bash examples/prompt_agent/run_gpt4o_agent.sh
```

# FAQ

## 1. Customize Memory Module
`verl-agent` supports a customizable and flexible memory system for managing and formatting interaction history between the agent and the environment. We provide a [SimpleMemory](./agent_system/memory/memory.py) implementation as a default starting point. This memory module is invoked within [env_manager.py](./agent_system/environments/env_manager.py) (i.e., `build_text_obs()`) to construct the observation at each step. 

Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.

## 2. Data Preparation
For most environments (e.g., AFLWorld, WebShop, Sokoban), we only use data preparation to indicate the modality, either "text" or "visual". For example, if the task is purely text-based, the data will just be an empty string "". If it involves visual input, it will be "\<image\>". As for agent input (including task instruction, observation and prompt), we follow the classical RL pipeline. That means the input of LLM agent comes from the environment's feedback through `env.step()`. In the case of search-r1 experiments where tasks are drawn from a dataset, we leverage the [env_kwargs](./examples/data_preprocess/preprocess_search_r1_dataset.py#L90) parameter to pass tasks into the environment, using: [envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))](./agent_system/multi_turn_rollout/rollout_loop.py#L301).

## 3. Customize Your Own Prompts  
We adopt a simple and minimal prompt format in our implementation. For example, in the WebShop environment:
```
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}. You are now at step {current_step} and your current observation is: {current_observation}. Your admissible actions of the current situation are: [{available_actions}].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```
If you wish to further enhance or customize them, you can find and edit them in: [agent_system/environments/prompts](./agent_system/environments/prompts/).


## 4. Add New Environments
To add a new environment, 
1. Create your environment package (gym-style interface and multi-process execution) in [agent_system/environments/env_package/](./agent_system/environments/env_package/)
2. Define the corresponding prompt files in [agent_system/environments/prompts](./agent_system/environments/prompts/). 
3. Register your new environment in [env_manager.py](./agent_system/environments/env_manager.py), following the structure defined by [EnvironmentManagerBase](./agent_system/environments/base.py#L19). 

For a reference implementation, see the webshop environment:
1. Environment package: [webshop package](./agent_system/environments/env_package/webshop)
2. Prompts: [webshop prompts](./agent_system/environments/prompts/webshop.py)
3. Environment Manager: [webshop env manager](./agent_system/environments/env_manager.py#L304)


# Contributing

We welcome and appreciate all contributions! If you have ideas to improve `verl-agent`, please feel free to submit a pull request (PR).

Example contributions include:
- **AppWorld Bug Fixes**: Fixed compatibility issues and ensured stable integration with the experimental AppWorld environment.
- **Asynchronous Rollout**: Improved training efficiency and throughput by supporting asynchronous rollout pipelines.
- **Additional Environments**: Added support for additional interactive environments to expand the benchmark coverage and task diversity.

# Acknowledgement

`verl-agent` codebase is built upon [veRL](https://github.com/volcengine/verl). 
The supported environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld). We extend our gratitude to the authors and contributors of these projects for their valuable work.

We would also like to thank the following contributors for their specific improvements to this project: WebShop bug fix ([@YSLIU627](https://github.com/YSLIU627)), GSPO support ([@MakeKJ](https://github.com/MakeKJ)), Qwen3-VL support ([@FabianSchuetze](https://github.com/FabianSchuetze)).

# Awesome Work Powered by verl-agent & GiGPO

- [GraphGPO](https://arxiv.org/abs/2605.26684): **Graph-Based Credit Assignment** for Agentic Reinforcement Learning.
- [HGPO](https://arxiv.org/pdf/2602.22817): **Hierarchy-of-Groups** Policy Optimization for Long-Horizon Agentic Tasks. 
- [Dr. MAS](https://arxiv.org/abs/2602.08847): Stable **end-to-end RL** post-training for **multi-agent LLM systems**. [![[code]](https://img.shields.io/github/stars/langfengQ/DrMAS)](https://github.com/langfengQ/DrMAS)
- [AgentOCR](https://arxiv.org/abs/2601.04786): Efficient token compression by rendering multi-turn agent history into images and adopting agentic self-compression.
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): An open-source framework for live-stream reinforcement learning tuning of LLM agents. [![[code]](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)](https://github.com/OpenManus/OpenManus-RL)
- [RLVMR](https://github.com/Tencent/DigitalHuman/tree/main/RLVMR): Providing agents with fine-grained meta-reasoning rewards in long-horizon tasks. [![[code]](https://img.shields.io/github/stars/Tencent/DigitalHuman)](https://github.com/Tencent/DigitalHuman/tree/main/RLVMR)
- [UI-S1](https://github.com/X-PLUG/MobileAgent/tree/main/UI-S1): A GUI automation model using semi-online reinforcement learning for stable long-horizon task execution. [![[code]](https://img.shields.io/github/stars/X-PLUG/MobileAgent)](https://github.com/X-PLUG/MobileAgent/tree/main/UI-S1)
- [Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558): A scalable, reward-free paradigm that bridges imitation learning and RL via implicit world modeling and self-reflection.
- [SPEAR](https://github.com/TencentYoutuResearch/SPEAR): **Self-imitation** with **Progressive Exploration** for Agentic Reinforcement Learning (ICLR 2026). [![[code]](https://img.shields.io/github/stars/TencentYoutuResearch/SPEAR)](https://github.com/TencentYoutuResearch/SPEAR/tree/main/)

# Citation
If you find `verl-agent` and `GiGPO` useful in your research or applications, we would appreciate it if you could cite our work:

```
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=langfengQ/verl-agent&type=Date)](https://www.star-history.com/#langfengQ/verl-agent&Date)
