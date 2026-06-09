# 目录架构说明

本文档根据当前代码目录整理，用于快速理解 `verl-agent` 仓库中各目录的职责边界。项目整体可以理解为三层：

- `verl/`：继承自 veRL 的核心训练框架，负责分布式训练、模型 worker、rollout、checkpoint、数据和指标等基础能力。
- `agent_system/`：面向 LLM Agent 的交互系统，负责 Agent baseline、环境封装、多轮 rollout、记忆、技能、状态、动作和行为分析。
- `recipe/`、`examples/`、`docs/`、`tests/`：分别承载论文/算法复现实验、可运行示例、使用文档和测试体系。

## 顶层目录

```text
verl-agent/
├── .github/              # GitHub 协作配置、CI 工作流、PR 模板和代码负责人配置
├── .vscode/              # VS Code 本地开发配置
├── agent_system/         # Agent、环境交互、多轮 rollout、记忆、技能、状态和行为分析
├── docker/               # Docker / Apptainer 镜像构建文件
├── docs/                 # Sphinx 文档、教程、API 说明和静态资源
├── examples/             # 训练、生成、数据预处理、服务启动等示例脚本
├── gigpo/                # GiGPO 核心算法实现
├── recipe/               # 特定算法或论文实验 recipe
├── scripts/              # 模型转换、合并、环境诊断和依赖安装脚本
├── tests/                # 单元测试、端到端测试、GPU/Ray/worker 等测试
└── verl/                 # veRL 核心包：训练器、worker、模型、工具和分布式控制
```

## 配置与工程目录

### `.github/`

GitHub 仓库维护目录，包含 PR 模板、CODEOWNERS、Dependabot 配置和 CI 工作流。

- `.github/workflows/`：正常启用的 GitHub Actions 工作流，例如文档构建、预提交检查、Ray 测试、E2E 测试、kernel 测试、vLLM/SGLang 测试等。
- `.github/workflows/disabled/`：暂时禁用或保留的工作流配置。

### `.vscode/`

VS Code 编辑器配置，目前主要用于统一本地开发设置。

### `docker/`

容器构建文件目录，覆盖不同硬件和推理后端：

- NVIDIA NGC / vLLM 镜像。
- SGLang 镜像。
- Megatron + vLLM + SGLang 镜像。
- ROCm / AMD 相关镜像。
- AWS EFA、SageMaker、veMLP 等部署场景镜像。
- `Apptainerfile.rocm`：面向 Apptainer/Singularity 的 ROCm 构建配置。

### `scripts/`

独立工具脚本目录：

- `diagnose.py`：运行环境诊断。
- `model_merger.py`：模型权重合并。
- `converter_hf_to_mcore.py`：Hugging Face checkpoint 到 Megatron-Core 格式转换。
- `install_vllm_sglang_mcore.sh`：安装 vLLM、SGLang、Megatron-Core 相关依赖。

## Agent 系统目录

### `agent_system/`

LLM Agent 训练相关的交互层。它把环境、记忆、多轮对话/行动循环和奖励管理连接到 `verl/` 的训练框架中。

新增的 agent infra 子模块用于搭建科研 baseline：`agents/` 定义不同 agent 策略，`memory/` 管理短期/长期记忆，`skills/` 管理 skill bank 与检索，`state/` 做显式状态跟踪，`action/` 做动作解析和校验，`prompt_builder/` 负责组合式 prompt 构造，`evaluation/` 负责行为分析和成本统计。

### `agent_system/agents/`

Agent baseline 抽象和实现目录。

- `base.py`：`BaseAgent` 和 `AgentOutput`，统一 action、prompt、thought 和 metadata 返回格式。
- `direct_agent.py`：Direct baseline，直接从观测生成动作。
- `react_agent.py`：ReAct baseline，显式保留 Thought/Action 通道。
- `reflexion_agent.py`：Reflexion baseline，用历史反思增强后续 prompt。
- `memory_agent.py`：Memory baseline，在构造 prompt 前检索长期记忆。
- `skill_agent.py`：Skill baseline，在构造 prompt 前检索可复用技能。

### `agent_system/environments/`

Agent 环境抽象与环境管理目录。

- `base.py`：环境基类/统一接口。
- `env_manager.py`：环境实例管理、并行环境和 group environment 组织。
- `prompts/`：各环境使用的 prompt 模板。
- `env_package/`：具体环境适配与内置/第三方环境源码。

### `agent_system/environments/prompts/`

按环境拆分的 prompt 定义目录：

- `alfworld.py`：ALFWorld 环境 prompt。
- `appworld.py`：AppWorld 环境 prompt。
- `gym_cards.py`：Gym Cards 环境 prompt。
- `search.py`：搜索/工具调用任务 prompt。
- `sokoban.py`：Sokoban 环境 prompt。
- `webshop.py`：WebShop 环境 prompt。

### `agent_system/environments/env_package/`

环境实现和适配层目录。每个子目录通常包含 `envs.py` 或 `projection.py`，用于把原始环境包装成项目统一使用的接口和观测/动作格式。

- `alfworld/`：ALFWorld 任务环境适配。
- `appworld/`：AppWorld 任务环境适配。
- `gym_cards/`：Gym Cards 视觉纸牌环境适配。
- `search/`：搜索工具调用任务环境适配。
- `sokoban/`：Sokoban 推箱子环境适配。
- `webshop/`：WebShop 电商网页交互环境适配。

### `agent_system/environments/env_package/alfworld/`

ALFWorld 环境封装。

- `alfworld/`：ALFWorld 原始/第三方环境源码。
- `configs/`：ALFWorld 运行配置，例如 TextWorld/Thor 配置。
- `projection.py`：将 ALFWorld 原始状态和动作投影到统一 agent 接口。

`alfworld/alfworld/` 下的目录主要来自 ALFWorld 环境本身：

- `agents/`：ALFWorld agent、controller、detector、expert、evaluation 和工具模块。
- `data/`：环境数据相关代码。
- `env/`：环境运行接口。
- `gen/`：任务生成、规划器、布局、图结构和脚本工具。

### `agent_system/environments/env_package/appworld/`

AppWorld 环境适配。

- `envs.py`：AppWorld 环境包装。
- `projection.py`：观测、动作、反馈格式转换。

### `agent_system/environments/env_package/gym_cards/`

Gym Cards 环境适配。

- `projection.py`：统一接口转换。
- `gym-cards/`：Gym Cards 原始/第三方环境源码、图片、字体和 notebook 示例。

### `agent_system/environments/env_package/search/`

搜索任务和工具调用环境适配。

- `envs.py`：搜索环境包装。
- `projection.py`：搜索观测/动作转换。
- `third_party/skyrl_gym/`：第三方 SkyRL Gym 风格环境与搜索工具实现。

### `agent_system/environments/env_package/sokoban/`

Sokoban 环境适配。

- `envs.py`：Sokoban 环境包装。
- `projection.py`：Sokoban 状态、图像/文本观测和动作转换。
- `sokoban/`：Sokoban 环境核心逻辑、房间生成和工具函数。

### `agent_system/environments/env_package/webshop/`

WebShop 环境适配。

- `envs.py`：WebShop 环境包装。
- `projection.py`：网页状态、动作和奖励反馈转换。
- `webshop/`：WebShop 原始/第三方环境源码。

`webshop/webshop/` 下的主要目录：

- `baseline_models/`：WebShop baseline 模型相关代码。
- `search_engine/`：商品搜索引擎逻辑。
- `transfer/`：数据/模型迁移相关脚本。
- `web_agent_site/`：WebShop 网页交互站点，包括模板、静态资源、属性、模型和环境接口。

### `agent_system/memory/`

Agent 记忆模块目录，用于控制每一步输入中包含哪些历史、状态摘要或上下文。

- `base.py`：记忆模块基础接口。
- `memory.py`：默认记忆实现。
- `trajectory_memory.py`：轨迹级短期记忆，适合保存 episode 内交互历史。
- `experience_memory.py`：成功经验长期记忆。
- `failure_memory.py`：失败经验长期记忆。
- `retriever.py`：记忆检索器，当前是轻量 keyword baseline，可替换为 embedding retriever。
- `storage.py`：JSONL 记忆持久化。
- `README.md`：记忆模块使用说明。

### `agent_system/skills/`

Skill bank 和 skill retrieval 目录。

- `base.py`：`Skill` 数据结构。
- `skill_bank.py`：skill 集合加载、保存和维护。
- `skill_retriever.py`：skill 检索器。
- `skill_evaluator.py`：统计 skill 使用后的收益。
- `skill_updater.py`：从轨迹中提议或更新 skill。
- `banks/`：不同任务的初始 skill bank，例如 ALFWorld、WebShop。

### `agent_system/state/`

显式状态跟踪目录，用于把原始 observation/action/info 转成可分析、可复用的结构化状态。

- `base.py`：状态跟踪器基础接口。
- `state_updater.py`：通用状态更新器。
- `alfworld_state_tracker.py`：ALFWorld 状态跟踪器。
- `webshop_state_tracker.py`：WebShop 状态跟踪器。

### `agent_system/action/`

动作解析、规范化和校验目录。

- `action_parser.py`：从模型输出中抽取可执行动作。
- `action_normalizer.py`：清理空白、统一动作格式。
- `action_validator.py`：基于候选动作或环境规则检查动作是否合法。

### `agent_system/prompt_builder/`

组合式 prompt 构造目录。

- `base.py`：prompt builder 基础接口。
- `react_prompt_builder.py`：ReAct prompt 构造。
- `memory_prompt_builder.py`：注入检索记忆的 prompt 构造。
- `skill_prompt_builder.py`：注入检索技能的 prompt 构造。

### `agent_system/evaluation/`

Agent 行为分析目录。

- `metrics.py`：success rate、平均步数等基础指标。
- `failure_analyzer.py`：失败类型统计。
- `memory_usage_analyzer.py`：记忆检索使用统计。
- `skill_usage_analyzer.py`：skill 使用统计。
- `cost_tracker.py`：prompt/completion token 成本统计。

### `agent_system/multi_turn_rollout/`

多轮 agent-environment 交互循环。

- `rollout_loop.py`：多轮 rollout 主循环。
- `utils.py`：rollout 辅助函数。

### `agent_system/reward_manager/`

Agent 交互过程的奖励管理。

- `episode.py`：按 episode 组织 reward、done、轨迹等信息。

## 核心训练框架目录

### `verl/`

项目的核心 Python 包，继承 veRL 的分布式 RLHF/RL 训练基础设施。

### `verl/trainer/`

训练入口、训练器和训练配置。

- `main_ppo.py`：PPO/GRPO/DAPO/RLOO 等训练主入口。
- `main_generation.py`：生成任务主入口。
- `main_eval.py`：评测主入口。
- `fsdp_sft_trainer.py`：FSDP SFT 训练器。
- `constants_ppo.py`：PPO 训练常量。
- `runtime_env.yaml`：Ray runtime 环境配置。
- `config/`：Hydra/YAML 训练配置。
- `ppo/`：PPO 系列算法核心逻辑、奖励处理、Ray trainer 和指标工具。

### `verl/trainer/config/`

训练配置文件目录：

- `ppo_trainer.yaml`：PPO/GRPO 等 RL 训练主配置。
- `ppo_megatron_trainer.yaml`：Megatron 后端训练配置。
- `sft_trainer.yaml`：SFT 训练配置。
- `generation.yaml`：批量生成配置。
- `evaluation.yaml`：评测配置。

### `verl/trainer/ppo/`

PPO 系列训练实现。

- `core_algos.py`：优势估计、loss、策略优化等核心算法。
- `ray_trainer.py`：基于 Ray 的分布式训练流程。
- `reward.py`：奖励计算与整合。
- `metric_utils.py`：训练指标统计工具。

### `verl/workers/`

分布式训练中的 worker 实现，负责 actor、critic、reward model、rollout 和参数切分/同步。

- `actor/`：策略模型 worker。
- `critic/`：价值模型 worker。
- `reward_manager/`：reward function / reward model 调度管理。
- `reward_model/`：奖励模型 worker，含 Megatron 版本。
- `rollout/`：推理 rollout worker。
- `sharding_manager/`：FSDP、vLLM、SGLang 等不同后端之间的参数切分和同步管理。

### `verl/workers/rollout/`

推理 rollout 后端目录。

- `naive/`：朴素/基础 rollout 实现。
- `sglang_rollout/`：SGLang 推理后端。
- `vllm_rollout/`：vLLM 推理后端。

### `verl/models/`

模型结构、checkpoint 转换和 Megatron 适配。

- `llama/`：LLaMA 系列模型适配。
- `qwen2/`：Qwen2/Qwen 系列模型适配。
- `mcore/`：Megatron-Core 相关模型桥接。
- `transformers/`：Hugging Face Transformers 相关模型工具。

`llama/megatron/` 和 `qwen2/megatron/` 下通常包含：

- `checkpoint_utils/`：checkpoint 加载、保存和格式转换。
- `layers/`：Megatron 模型层实现或适配。

### `verl/single_controller/`

单控制器分布式抽象，用于统一管理多个 worker。

- `base/`：worker、worker group、装饰器和注册中心等基础抽象。
- `base/megatron/`：Megatron worker 和 worker group 抽象。
- `base/register_center/`：注册中心，目前包含 Ray 实现。
- `ray/`：Ray backend 的控制器实现。

### `verl/utils/`

通用工具目录。

- `checkpoint/`：checkpoint 保存、加载、转换、合并等工具。
- `dataset/`：数据集、采样和 batch 处理工具。
- `debug/`：调试工具。
- `experimental/`：实验性工具或功能。
- `logger/`：日志和 WandB 等记录器。
- `megatron/`：Megatron 相关工具。
- `metric/`：指标计算与聚合。
- `rendezvous/`：分布式 rendezvous/初始化辅助。
- `reward_score/`：内置 reward function 和评测打分逻辑。

### `verl/utils/reward_score/`

奖励/打分函数集合。

- `prime_code/`：PRIME 代码任务打分相关逻辑。
- `prime_math/`：PRIME 数学任务打分相关逻辑。
- `sandbox_fusion/`：Sandbox Fusion 代码执行/打分集成。

### `verl/tools/`

工具调用相关基础设施。

- `utils/`：工具配置、schema、调用结果处理等辅助逻辑。

### `verl/third_party/`

与第三方推理后端或外部实现兼容的适配代码。

- `sglang/`：SGLang 相关兼容层。
- `vllm/`：vLLM 相关兼容层。
- `vllm/vllm_v_0_5_4/`、`vllm/vllm_v_0_6_3/`：针对不同 vLLM 版本的兼容实现。

### `verl/version/`

版本文件目录，`pyproject.toml` 和 `setup.py` 会从这里读取包版本。

## 算法与实验目录

### `gigpo/`

GiGPO 算法核心实现目录。

- `core_gigpo.py`：Group-in-Group Policy Optimization 的核心计算逻辑。

### `recipe/`

论文方法、算法变体和实验复现配置目录。通常每个子目录包含入口脚本、Ray trainer/worker、核心算法、YAML 配置和运行脚本。

- `GraphGPO/`：GraphGPO 实验 recipe，包含配置、训练脚本和可视化产物。
- `dapo/`：DAPO recipe。
- `hgpo/`：HGPO recipe，包含 HGPO trainer、环境管理、核心算法、配置和运行脚本。
- `memory_skill/`：Memory/Skill 正式实验 recipe 预留目录。
- `prime/`：PRIME recipe，包含 PRIME 数据/奖励模型、trainer、worker 和配置。
- `r1/`：R1 distill/eval 相关 recipe，包含任务定义、数据处理和评测入口。
- `spin/`：SPIN recipe，包含 trainer、worker、核心算法和配置。
- `sppo/`：SPPO recipe。

### `recipe/*/config/`

各 recipe 的 Hydra/YAML 配置目录，定义模型路径、数据、算法参数、环境参数、rollout 参数和分布式训练参数。

### `recipe/r1/tasks/`

R1 评测任务定义目录，例如 math、GPQA、LiveCodeBench 等。

### `recipe/GraphGPO/vis_*`

GraphGPO 相关可视化结果目录，主要存放 PDF 轨迹、对比图或实验展示产物。

## 示例目录

### `examples/`

面向用户的可运行示例脚本目录。

- `dapo_trainer/`：DAPO 训练示例。
- `data_preprocess/`：数据预处理脚本，如 GSM8K、MATH、AIME、Search-R1、多轮工具调用数据等。
- `env_server/`：外部环境服务启动脚本，例如 AppWorld server。
- `generation/`：批量生成/推理示例。
- `gigpo_dynamic_trainer/`：动态采样 GiGPO 训练示例。
- `gigpo_trainer/`：GiGPO 训练示例，覆盖 ALFWorld、WebShop、Search、Sokoban、Blackjack、NumberLine 等任务。
- `grpo_trainer/`：GRPO 训练示例。
- `gspo_trainer/`：GSPO 训练示例。
- `memory_skill_agent/`：Memory/Skill prompt-based baseline 示例入口。
- `ppo_trainer/`：PPO 训练示例。
- `prompt_agent/`：基于 GPT-4o 等外部模型的 prompt agent 示例。
- `ray/`：Ray 使用教程或 notebook。
- `rloo_trainer/`：RLOO 训练示例。
- `search/`：搜索任务相关示例。
- `sft/`：监督微调示例。
- `slurm/`：Slurm 集群启动 Ray/训练任务的示例。
- `split_placement/`：模型/worker 分离部署或 placement 示例。

### `examples/search/retriever/`

搜索任务检索服务相关代码和启动脚本。

### `examples/sft/gsm8k/`

GSM8K SFT 训练脚本，覆盖不同模型、PEFT、sequence parallel、Liger 等配置。

### `examples/sft/multiturn/`

多轮数据/对话形式的 SFT 示例。

### `examples/split_placement/config/`

split placement 示例的训练配置。

## 文档目录

### `docs/`

Sphinx 文档源码目录。

- `_static/`：文档静态资源。
- `advance/`：高级主题，如 checkpoint、FSDP 扩展、Megatron 扩展、DPO 扩展、LoRA、placement、RoPE 等。
- `algo/`：算法说明，如 PPO、GRPO、DAPO、SPIN、SPPO、baseline。
- `amd_tutorial/`：AMD/ROCm/vLLM 构建和使用教程。
- `api/`：API 文档入口。
- `ascend_tutorial/`：Ascend/NPU 快速开始文档。
- `examples/`：示例讲解，如 PPO 代码架构、GSM8K、多模态、Sandbox Fusion。
- `faq/`：常见问题。
- `gigpo/`：GiGPO 相关图片资源和说明素材。
- `agent_infra/`：基于本项目搭建科研 agent infra 的参考文档。
- `perf/`：性能调优文档。
- `preparation/`：数据准备和 reward function 准备文档。
- `sglang_multiturn/`：SGLang 多轮、工具调用、Sandbox Fusion 示例文档。
- `start/`：安装、快速开始、多机和 Ray debug 教程。
- `workers/`：FSDP、Megatron、Ray trainer、SGLang worker 等 worker 文档。

### `docs/_static/`

文档静态资源目录。

- `js/`：文档页面 JavaScript 资源。

## 测试目录

### `tests/`

测试代码目录，覆盖算法、训练器、worker、Ray、GPU/NPU、reward score、sandbox 和端到端流程。

- `distributed/`：分布式行为测试。
- `e2e/`：端到端训练/生成/评测脚本和测试数据。
- `gpu_utility/`：GPU 工具测试。
- `kernels/`：kernel 相关测试。
- `models/`：模型适配测试。
- `npu/`：NPU/Ascend 相关测试。
- `ray_cpu/`：Ray CPU 场景测试。
- `ray_gpu/`：Ray GPU 场景测试。
- `reward_score/`：reward score 测试。
- `sandbox/`：沙箱执行测试。
- `sanity/`：基础健康检查。
- `single_controller/`：single controller 抽象测试。
- `agent_system/`：agent infra 中 memory、skill、state、action、prompt builder 等模块测试。
- `trainer/`：训练器测试。
- `utils/`：工具函数测试。
- `workers/`：worker 和 rollout 测试。

### `tests/e2e/`

端到端测试目录，包含可执行 shell 脚本、任务环境和小型测试资源。

- `arithmetic_sequence/`：算术序列端到端任务，包含数据、模型和 RL 配置。
- `envs/`：E2E 测试环境。
- `generation/`：生成流程测试。
- `ppo_trainer/`：PPO trainer E2E 测试。
- `sft/`：SFT E2E 测试。

### `tests/e2e/envs/digit_completion/`

digit completion 测试任务环境，包含 tokenizer 和 task 定义。

### `tests/ray_cpu/`

Ray CPU 测试。

- `check_worker_alive/`：检查 Ray worker 生命周期和存活状态。

### `tests/ray_gpu/`

Ray GPU 测试。

- `detached_worker/`：detached worker 行为测试。

### `tests/single_controller/base/`

single controller 基础抽象测试。

### `tests/trainer/ppo/`

PPO trainer 和 PPO 指标工具测试。

### `tests/utils/`

通用工具测试。

- `cpu_tests/`：CPU 可运行工具测试。
- `gpu_tests/`：GPU 相关工具测试。
- `gpu_tests/checkpoint/`：checkpoint GPU 测试。
- `gpu_tests/dataset/`：dataset GPU 测试。
- `gpu_tests/megatron/`：Megatron GPU 测试。

### `tests/workers/rollout/`

rollout worker 测试。

- `resource/`：rollout 测试资源。
- `resource/tool_configs/`：工具调用配置测试资源。

## 根目录重要文件

- `README.md`：项目总说明、安装、功能、结果和运行示例。
- `LICENSE`：Apache 2.0 许可证。
- `Notice.txt`：版权和第三方声明。
- `pyproject.toml`：PEP 621 项目元数据、构建配置和 Ruff 配置。
- `setup.py`：兼容旧安装方式的 setuptools 配置。
- `requirements.txt`：基础依赖。
- `requirements_sglang.txt`：SGLang 相关依赖。
- `requirements-npu.txt`：NPU/Ascend 相关依赖。
- `.pre-commit-config.yaml`：pre-commit 检查配置。
- `.readthedocs.yaml`：Read the Docs 构建配置。
- `.gitignore`：Git 忽略规则。

## 常见开发入口

- 训练入口：`verl/trainer/main_ppo.py`
- 评测入口：`verl/trainer/main_eval.py`
- 生成入口：`verl/trainer/main_generation.py`
- PPO/GRPO 核心算法：`verl/trainer/ppo/core_algos.py`
- GiGPO 核心算法：`gigpo/core_gigpo.py`
- 多轮 Agent rollout：`agent_system/multi_turn_rollout/rollout_loop.py`
- 环境管理：`agent_system/environments/env_manager.py`
- 训练主配置：`verl/trainer/config/ppo_trainer.yaml`
- Agent 训练示例：`examples/gigpo_trainer/`、`examples/grpo_trainer/`、`examples/ppo_trainer/`
