# 快速开始

在昇腾 NPU 上从源码构建并部署 Text Generation Inference（TGI）推理服务，
对 Qwen3-0.6B 完成端到端文本生成：单卡基线 + 双卡 HCCL 张量并行（`--num-shard 2`）。

本文档覆盖上游官方 release 中**尚未包含**的 Ascend NPU 支持：构建、安装、
启动与推理验证使用
[cosdt/text-generation-inference](https://github.com/cosdt/text-generation-inference)
（TGI 官方仓库的 fork，Ascend 适配以 release 形式发布在该 fork 上）。

## 前置条件

### 硬件

Atlas 900 A2 PODc（Ascend 910B × 2），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 3.12 环境
- 可用的 CANN 9.1.0（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- Rust 工具链（本文档「安装 Rust 工具链」小节会通过 rustup 安装，无需提前准备）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| transformers | 4.57.6 |
| kernels | 0.5.0 |
| 模型 | [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) |

## 1. 检查环境

### 确认 NPU 设备

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+---------------------------+---------------+----------------------------------------------------+
| No running processes found in NPU 0                                                            |
+---------------------------+---------------+----------------------------------------------------+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
> 本文档的双卡验证需要**至少两张卡**可见。

### 检查 Python 版本

```shell #test id="check-python"
python --version
```

输出结果如下：

```shell #test-result id="check-python" fuzzy='xxx'
Python 3.12.xxx
```

### 检查 NPU 设备运行时可用

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 2. 安装依赖与工具链

### 系统依赖

TGI 是 Rust + Python 双栈项目：Rust 侧编译需要 C/C++ 工具链与 protobuf 编译器，
PyO3 嵌入 Python 需要开发头文件：

```shell #test-setup
apt-get update -qq
apt-get install -y -qq build-essential protobuf-compiler pkg-config libssl-dev curl git python3-dev
```

### Python 依赖

torch / torch_npu 从华为昇腾源安装（与 CANN 9.1.0 匹配的 2.9 系列），其余
Python 依赖用 uv 安装：

```shell #test-setup
python -m pip install -q uv
uv pip install \
  --index-url https://repo.huaweicloud.com/ascend/repos/pypi/simple \
  "torch==2.9.0" "torch_npu==2.9.0.post2"
uv pip install \
  "transformers==4.57.6" "accelerate==1.15.0" "modelscope==1.37.0" \
  "kernels==0.5.0" "grpcio-tools>=1.69.0" "mypy-protobuf>=3.6.0"
```

> `kernels` 必须固定 0.5.0：它是 TGI Python server 的构建后端，0.5.0 自带
> `kernels.lockfile`，构建时不会去下载 CUDA 专属内核；更新的版本缺该文件，
> 在无 CUDA 的 aarch64 昇腾机器上会构建失败。

### 安装 Rust 工具链

```shell #test-setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain 1.85.1 --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
```

> TGI 仓库根目录的 `rust-toolchain.toml` 固定 1.85.1，因此这里直接安装该版本。

## 3. 下载基础模型

默认使用 **ModelScope** 下载 Qwen3-0.6B（约 1.2 GB，落到默认缓存
`~/.cache/modelscope`）：

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))" | tail -n 1
```

## 4. 构建并安装 TGI

### 获取源码

克隆 fork 并 checkout 到本次看护的 release tag（工作流注入 `UPSTREAM_REF`，
`<ref>` 为该 tag）：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch "<ref>" \
  https://github.com/cosdt/text-generation-inference.git tgi \
  || { git clone --depth 1 \
         https://github.com/cosdt/text-generation-inference.git tgi \
       && git -C tgi fetch --depth 1 origin "<ref>" \
       && git -C tgi checkout -q FETCH_HEAD; }
```

> `<ref>` 为要安装的 release tag（也可替换为任意分支名或 commit SHA）。
> 手工执行时无需 `UPSTREAM_REF`，直接把 `<ref>` 换成 tag，如 `v3.3.7-npu`。

### 编译 Rust 二进制

编译 launcher 与 router（`--profile release-opt` 为上游提供的发布优化 profile）。
PyO3 嵌入 Python 需要 `PYO3_PYTHON` 指向环境里的 Python，protobuf 代码生成
需要 `PROTOC`：

```shell #test-setup
cd tgi
export PYO3_PYTHON="$(command -v python)"
export PROTOC="$(command -v protoc)"
cargo build --profile release-opt \
  -p text-generation-launcher -p text-generation-router-v3
```

### 安装 Python server

```shell #test-setup
cd tgi
uv pip install --no-build-isolation -e server
make -C server gen-server-raw
```

### 检查构建产物

```shell #test id="check-build"
$PWD/tgi/target/release-opt/text-generation-launcher --version
python -c "import text_generation_server; print('server import ok')"
```

输出结果如下：

```shell #test-result id="check-build" fuzzy='xxx'
text-generation-launcher xxx
server import ok
```

## 5. 启动服务并验证推理

服务生命周期由 fork 仓库自带的 `start-tgi.sh` / `stop-tgi.sh` 脚本管理。
下面按步骤演示，最后一节给出把全部步骤串起来的一键端到端验证（CI 看护
执行的正是这段流程）。以下命令均在「获取源码」克隆出的 `tgi` 目录下执行。

### 5.1 启动服务

`start-tgi.sh` 负责后台启动、就绪轮询与日志落盘（默认 `/tmp/tgi.log`），
脚本返回 `[READY]` 即服务可用。单卡：

```shell
cd tgi
./start-tgi.sh --num-shard 1 --devices 0
```

双卡 HCCL 张量并行：

```shell
cd tgi
./start-tgi.sh --num-shard 2 --devices 0,1
```

> 不传 `--model-id` 时默认使用 Qwen/Qwen3-0.6B，首次启动自动经 ModelScope
> 下载（缓存于 `~/.cache/modelscope`，之后直接命中）；也可用
> `--model-id /path/to/model` 指定本地路径。

启动成功的输出：

```
[START] model_id=/path/to/model
[START] num_shard=1  devices=0  port=8080
[START] launcher pid 12345
[READY] TGI is serving on http://127.0.0.1:8080 after ~130s
[READY] stop with: ./stop-tgi.sh
```

**冷启动约 2-3 分钟是正常现象**，`[READY]` 出现前服务不可访问，耗时主要在：

- 模型权重加载并搬运到 NPU 显存（Qwen3-0.6B 约 1.2 GB）；
- KV cache 分配与模型预热（`Warming up model` 阶段）；
- 双卡时还需 HCCL 组网初始化，比单卡略久。

期间可另开终端观察 `tail -f /tmp/tgi.log`——出现 `Connected` 即 HTTP 服务
已就绪（日志时间跨度即权重加载与预热耗时）：

```
2026-09-21T06:20:42Z  INFO text_generation_router_v3: Warming up model
2026-09-21T06:21:37Z  INFO text_generation_launcher: KV-cache blocks: 2708, size: 64
2026-09-21T06:21:47Z  INFO text_generation_router::server: Connected
```

随后 `start-tgi.sh` 的就绪轮询打印 `[READY] TGI is serving on
http://127.0.0.1:8080 after ~130s`——~130s 为 910B 实测冷启动时长，不同
机器略有差异，属正常范围。

### 5.2 确认服务就绪

```shell
curl -4s http://127.0.0.1:8080/info
```

返回模型信息 JSON（含 `model_id`、`max_total_tokens` 等字段）即就绪；
为空则说明服务尚未就绪或已退出，用 `tail -50 /tmp/tgi.log` 查看原因。

### 5.3 发起推理

```shell
curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))'
```

`do_sample=false` 为贪心解码，相同输入输出确定。示例输出：

```
2  The question is: What is the sum of the numbers 1
```

OpenAI 兼容接口：`curl -4 http://127.0.0.1:8080/v1/chat/completions`。

### 5.4 停止服务

```shell
cd tgi
./stop-tgi.sh
```

脚本会优雅退出 launcher 并确认 NPU 上无残留进程；若 `npu-smi info` 里
仍有残留（残留进程会占用 NPU 与端口，导致新实例报
`EJ0003 Failed to bind the IP port`），用 `./stop-tgi.sh --force` 清理。

### 5.5 一键端到端验证（单卡基线 + 双卡张量并行）

下面的两个块把上述步骤串成一次完整验证，可直接整块复制执行，也是 CI
看护执行的流程。先跑单卡基线：启动、调用 `/generate` 做一次真实推理、
停止，回复作为双卡验证的基线（此块为标准输出捕获块，回复存入
`reply_single`，脚本日志重定向到 stderr 不参与捕获）：

```shell #test-setup store="reply_single" load="model_path>>model_path"
set -e
cd tgi
cleanup() {
  ./stop-tgi.sh --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
./start-tgi.sh --num-shard 1 --devices 0 --model-id "<model_path>" >&2
curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))'
./stop-tgi.sh >&2
```

再用两张卡（`--num-shard 2`，HCCL 张量并行）跑一次相同请求，验证回复
非空、且与单卡基线**逐字一致**（贪心解码下张量并行不改变输出）：

```shell #test id="smoke-tp2" load="model_path>>model_path" load="reply_single>>reply_single"
set -e
cd tgi
cleanup() {
  ./stop-tgi.sh --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# wait until the single-card service above has fully released port 8080
for i in $(seq 1 30); do
  curl -4fs http://127.0.0.1:8080/info >/dev/null 2>&1 || break
  sleep 2
done

./start-tgi.sh --num-shard 2 --devices 0,1 --model-id "<model_path>" >&2
REPLY=$(curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))')
[ -n "$REPLY" ] || { echo "empty reply"; exit 1; }
[ "$REPLY" = "<reply_single>" ] || { echo "TP2 reply differs from single-card baseline"; echo "single: <reply_single>"; echo "tp2:    $REPLY"; exit 1; }
echo "TGI-TP2-OK: $REPLY"
./stop-tgi.sh >&2
```

输出结果如下（`...` 为模型回复内容，与单卡基线一致）：

```shell #test-result id="smoke-tp2"
TGI-TP2-OK: ...
```

## 小贴士

- **更多卡**：`--num-shard N` 配合 `ASCEND_VISIBLE_DEVICES=0,1,...,N-1` 即可
  做 N 卡 HCCL 张量并行（双卡为本文档看护范围）。
- **输出一致性对比**：`do_sample=false` 贪心解码下，相同模型与参数的输出是
  确定的——双卡验证正是用它断言张量并行不改变输出。
- **常用查询**：`curl -4 http://127.0.0.1:8080/info` 查看服务信息，
  `curl -4 http://127.0.0.1:8080/v1/chat/completions` 走 OpenAI 兼容接口。
