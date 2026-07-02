# 环境激活说明

## ALFWorld 环境

```bash
conda activate verl-agent-alfworld
```

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| alfworld | 0.4.2 |
| gymnasium | 0.29.1 |
| textworld | 1.7.0 |
| 数据路径 | `~/.cache/alfworld/` |

验证安装：
```bash
python -c "import alfworld; import gymnasium; import textworld; print('OK')"
```

玩一把 Textworld 游戏：
```bash
alfworld-play-tw
```

---

## WebShop 环境

```bash
conda activate verl-agent-webshop
export JAVA_HOME="/opt/homebrew/Cellar/openjdk@17/17.0.19/libexec/openjdk.jdk/Contents/Home"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
export PATH="$JAVA_HOME/bin:$PATH"
```

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| gym | 0.24.0 |
| pyserini | 0.17.0 |
| Java | OpenJDK 17 |

验证安装：
```bash
python -c "import gym; import pyserini; print('OK')"
```

> 💡 可以把 Java 环境变量写入 `~/.zshrc`：
> ```bash
> echo 'export JAVA_HOME="/opt/homebrew/Cellar/openjdk@17/17.0.19/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc
> echo 'export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"' >> ~/.zshrc
> echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
> ```
