# run/ — 独立 Agent-Environment 运行入口

通过 **YAML 配置文件** 组装 Agent + Environment + LLM，在 benchmark 环境中独立运行和评估 Agent。

---

## 快速开始

```bash
# 1. 激活环境
conda activate verl-agent-alfworld

# 2. 安装依赖
pip install -r run/requirements.txt

# 3. 设置 LLM API Key
export OPENAI_API_KEY="sk-..."

# 4. 运行
python -m run.run --config run/configs/alfworld_react.yaml
```

输出结果保存在 `run/outputs/` 目录下。

---

## 设计思路

```
  ┌─────────────────── YAML 配置 ───────────────────┐
  │  agent_type: "react"                             │
  │  env.name: "alfworld"                            │
  │  llm: { provider: "openai", model: "gpt-4o" }    │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
  ┌─────────────────── Runner ──────────────────────┐
  │                                                  │
  │  LLM Client ──► Agent ──► Environment            │
  │      │                        │                  │
  │      └──── 交互循环 ──────────┘                  │
  │                                                  │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
              run/outputs/xxx.json
```

**切换组合只需换配置文件，零代码修改。**

---

## 命令行

```bash
python -m run.run --config <config.yaml> [选项]
```

| 参数 | 说明 |
|------|------|
| `--config`, `-c` | **必填**。YAML 配置文件路径 |
| `--episodes`, `-n` | 覆盖配置中的 `num_episodes` |
| `--verbose`, `-v` | 打印每步详细信息 |
| `--output-dir`, `-o` | 结果输出目录（默认 `run/outputs/`） |

示例：

```bash
# 跑 5 个 episode
python -m run.run -c run/configs/alfworld_react.yaml -n 5

# 安静模式 + 自定义输出目录
python -m run.run -c run/configs/webshop_react.yaml -o results/ -v
```

---

## 配置文件

配置文件是 YAML 格式，包含 4 个部分：

```yaml
name: "my_experiment"          # 实验名称

agent_type: "react"            # 见下方"支持的 Agent"

llm:                           # LLM API 配置
  provider: "openai"           # openai | anthropic
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}" # ${ENV_VAR} 从环境变量读取
  temperature: 0.0
  max_tokens: 512
  # base_url: "https://..."    # 自定义 API 端点（vLLM / Ollama 等）

env:                           # 环境配置
  name: "alfworld"             # 见下方"支持的环境"
  history_length: 10           # prompt 中包含的历史步数
  max_steps: 50                # 单 episode 最大步数
  seed: 42
  kwargs:                      # 环境专属参数
    eval_dataset: "eval_in_distribution"

num_episodes: 1                # 运行 episode 数
verbose: true                  # 是否打印每步信息
```

### 环境变量解析

`${ENV_VAR}` 格式的值会自动从环境变量中读取：

```yaml
api_key: "${OPENAI_API_KEY}"          # → os.environ["OPENAI_API_KEY"]
base_url: "${CUSTOM_LLM_ENDPOINT}"    # → os.environ["CUSTOM_LLM_ENDPOINT"]
```

---

## 支持的 Agent

| agent_type | 类 | 说明 |
|------------|-----|------|
| `react` | `ReActAgent` | Thought/Action 通道，从 env prompt 直接生成动作 |
| `reflexion` | `ReflexionAgent` | 在 prompt 前追加历史反思 |
| `memory` | `MemoryAgent` | 检索长期记忆注入 prompt |
| `skill` | `SkillAgent` | 检索可复用技能注入 prompt |
| `direct` | `DirectAgent` | 最简基线，直接从观测生成动作 |

---

## 支持的环境

### ALFWorld

具身 AI 文本交互任务（pick & place, heat, cool, clean, examine 等）。

```bash
conda activate verl-agent-alfworld
python -m run.run -c run/configs/alfworld_react.yaml
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `env.name` | `alfworld` | |
| `kwargs.eval_dataset` | `eval_in_distribution` | 分布内评估 |
| | `eval_out_of_distribution` | 分布外评估 |

### WebShop

电商网页交互任务（搜索、点击、选择属性、购买）。

```bash
conda activate verl-agent-webshop
# 确保 Java 环境变量已设置
export JAVA_HOME="/opt/homebrew/Cellar/openjdk@17/17.0.19/libexec/openjdk.jdk/Contents/Home"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
export PATH="$JAVA_HOME/bin:$PATH"

python -m run.run -c run/configs/webshop_react.yaml
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `env.name` | `webshop` | |
| `kwargs.use_small` | `true` | 1000 商品小数据集（测试用） |
| | `false` | 全量数据集 |
| `kwargs.human_goals` | `true` | 人类撰写的目标指令 |

> ⚠️ WebShop 数据文件需提前下载：
> ```bash
> cd agent_system/environments/env_package/webshop/webshop
> bash setup.sh -d small
> ```

### Search · Sokoban · Gym Cards · AppWorld

环境代码已就绪，按需创建配置文件即可。示例：

```yaml
# run/configs/sokoban_react.yaml
name: "sokoban_react"
agent_type: "react"
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
env:
  name: "sokoban"
  max_steps: 100
  kwargs:
    dim_room: [7, 7]
    num_boxes: 2
    mode: "tiny_rgb_array"
```

---

## LLM 后端

### OpenAI（及兼容 API）

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
```

支持所有 OpenAI 兼容端点，包括：
- **vLLM** 自部署：设置 `base_url: "http://localhost:8000/v1"`
- **Ollama**：设置 `base_url: "http://localhost:11434/v1"`
- **各类中转 API**

### Anthropic

```yaml
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  api_key: "${ANTHROPIC_API_KEY}"
```

---

## 输出格式

结果保存为 JSON 文件 `run/outputs/<name>_<timestamp>.json`：

```json
{
  "config": "run/configs/alfworld_react.yaml",
  "name": "alfworld_react",
  "agent_type": "react",
  "env_name": "alfworld",
  "llm_model": "gpt-4o",
  "num_episodes": 5,
  "successes": 3,
  "success_rate": 0.6,
  "avg_steps": 12.4,
  "results": [
    {
      "episode": 0,
      "success": true,
      "total_reward": 10.0,
      "num_steps": 8,
      "elapsed_sec": 23.5
    }
  ],
  "trajectories": [
    [
      {
        "step": 0,
        "observation": "You are in the middle of a room...",
        "model_output": "<think>...</think><action>goto cabinet</action>",
        "action": "goto cabinet",
        "reward": 0.0,
        "done": false
      }
    ]
  ]
}
```

---

## 添加新 Agent

1. 在 `agent_system/agents/` 中实现，继承 `BaseAgent`
2. 在 `run/runner.py` 的 `_create_agent()` 中注册

```python
elif agent_type == "my_agent":
    return MyAgent(prompt_builder=..., action_parser=action_parser)
```

3. 使用时在 YAML 中设置 `agent_type: "my_agent"`

## 添加新环境

1. 在 `agent_system/environments/env_package/` 中实现 builder + projection
2. 在 `run/runner.py` 的 `_create_env()` 和 `_env_config_to_omega()` 中注册
3. 使用时在 YAML 中设置 `env.name: "my_env"`

---

## 文件索引

| 文件 | 职责 |
|------|------|
| [config.py](config.py) | `ExperimentConfig` 等 dataclass、YAML 加载、OmegaConf 适配 |
| [llm_client.py](llm_client.py) | `BaseLLMClient` → `OpenAIClient` / `AnthropicClient` |
| [runner.py](runner.py) | `Runner` 类：环境创建、Agent 创建、交互循环 |
| [run.py](run.py) | CLI 入口（argparse） |
| [configs/](configs/) | YAML 配置文件 |
| [outputs/](outputs/) | 运行结果 |
