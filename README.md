# gguf — GGUF 解析 + LLM 推理框架

一个纯 Rust 实现，覆盖 GGUF（GGML Universal Format，llama.cpp 生态）的**元数据解析**与**大模型推理**：

- **解析层**：读取文件头、KV 键值元数据、张量描述符；只读元数据区不加载权重数据体，可高效处理数十 GB 文件
- **推理层**：加载 GGUF 模型执行 LLM 文本生成，支持 llama / qwen2 / mistral 架构、BPE 分词、温度/Top-K/Top-P/Min-P/Repeat Penalty 采样、KV Cache、GQA Attention、RoPE、Q4_0~Q8_K 量化反量化

同时提供：

- **`gguf` 库 crate**：可复用的解析 + 推理 API
- **`gguf-dump` CLI 工具**：查看任意 GGUF 文件的元数据，支持文本与 JSON 输出、词表查询
- **`gguf-infer` CLI 工具**：加载模型执行文本生成（流式输出、交互问答）
- **`gguf-gui` 桌面 GUI**：基于 egui 的图形界面，提供对话 / Prompt 补全 / 模型信息三大标签页

## 特性

### 元数据解析

- 完整解析 GGUF version 3 文件头 + 全部 **13 种** `gguf_type`（含数组）
- 解析张量描述符（名称 / 形状 / 数据类型 / 数据偏移 / 元素数 / 估算大小）
- **零权重加载**：仅读取元数据区（通常 < 几十 MB），适合大文件
- `from_path` 优先 **mmap** 零拷贝，失败自动回退普通读取
- 健壮的错误处理：损坏 / 截断 / 非法 UTF-8 / 大数组 count 均返回 `Err`，**不 panic**
- 未知张量 dtype 优雅降级为 `GgmlType::Unknown(n)`
- `gguf-dump` CLI 支持文本摘要、KV 列表、张量表格、`--json`（大数组自动截断）、词表查询（`-T` / `-S` / `--merge-contains`）

### LLM 推理

- 支持 **llama / qwen2 / mistral** 架构（`LlamaModel` forward）
- **BPE 分词器**：从 GGUF KV 中加载 `tokenizer.ggml.tokens` + `merges`，encode / decode
- **采样器**：温度 → softmax → Top-K → Top-P → Min-P → 随机；Repeat Penalty；贪心解码（`temperature=0`）；可复现种子
- **KV Cache**：增量推理，prompt 预填充 + 单 token 逐次 forward
- **GQA Attention**：Grouped Query Attention（`n_kv_heads < n_heads`）
- **RoPE**：旋转位置编码
- **量化反量化**：Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q2_K / Q3_K_L / Q3_K_M / Q3_K_S / Q4_K / Q5_K / Q6_K / Q8_K / F16 / BF16
- **`gguf-infer` CLI**：流式逐 token 输出、交互问答模式（`--chat`），参数化采样，退出码映射错误类型
- **`gguf-gui` 桌面 GUI**：egui 图形界面，三大标签页（对话 / Prompt 补全 / 模型信息），后台推理线程 + mpsc 消息通道，支持停止生成 / 重置上下文 / 词表查询 / JSON 导出

## 项目结构

```
model_study/
├── Cargo.toml              # 包定义（features: mmap / json / parallel / gui）
├── src/
│   ├── lib.rs              # gguf 库入口，re-export 公共类型
│   ├── types.rs            # GgufType, GgmlType, GgufValue, GgufArray
│   ├── tensor.rs           # TensorInfo
│   ├── header.rs           # GgufHeader
│   ├── file.rs             # GgufFile（核心解析）
│   ├── cursor.rs           # Cursor 字节读取器（内部，带边界保护）
│   ├── error.rs            # GgufError
│   ├── console_wide.rs     # Windows 宽字符控制台 I/O（ReadConsoleW/WriteConsoleW）
│   ├── infer/              # 推理模块
│   │   ├── mod.rs          # 子模块入口，re-export Engine
│   │   ├── engine.rs       # Engine（组合模型+分词器+采样器，generate/complete/chat）
│   │   ├── tokenizer.rs    # BPE 分词器（encode/decode，从 GGUF KV 加载）
│   │   ├── sampler.rs      # 采样器（温度/Top-K/Top-P/Min-P/Repeat Penalty）
│   │   ├── quant.rs        # 量化反量化（Q4_0~Q8_K/F16/BF16）+ 量化 matvec
│   │   ├── cache.rs        # KV Cache（增量推理）
│   │   ├── ops.rs          # 基础算子（softmax、layer_norm、matmul 等）
│   │   └── model/
│   │       ├── mod.rs      # 模型模块入口
│   │       ├── hparams.rs  # 超参数解析（embed_dim/n_heads/n_kv_heads/ffn_dim 等）
│   │       └── llama.rs    # LlamaModel forward（Attention/FFN/RoPE/GQA）
│   └── bin/
│       ├── gguf_dump.rs    # 元数据 CLI（含词表查询）
│       ├── gguf_infer.rs   # 推理 CLI（含交互问答模式）
│       ├── gguf_gui.rs     # GUI 入口（egui/eframe，需 --features gui）
│       └── gui/            # GUI 子模块
│           ├── app.rs      # 主 App + 标签页 + 工具栏/状态栏
│           ├── chat_view.rs# 对话视图（消息列表）
│           ├── prompt_view.rs# Prompt 补全视图
│           ├── model_view.rs# 模型信息视图（KV/张量/词表查询/JSON 导出）
│           ├── settings.rs # 采样参数设置面板
│           ├── state.rs    # GUI 状态与数据结构
│           └── inference.rs# 后台推理线程与消息通道
├── tests/
│   ├── parse_buffer.rs     # 解析测试（覆盖 13 种 KV 类型）
│   ├── cli.rs              # gguf-dump CLI 集成测试
│   ├── inference.rs        # gguf-infer 端到端集成测试
│   └── common/             # 测试工具（构造内存 GGUF 缓冲）
├── examples/
│   └── f16_debug.rs        # F16 调试示例
├── spec.md                 # 技术规范
├── checklist.md            # 验收清单
└── tasks.md                # 任务拆解
```

## 构建

```bash
# 默认开启 mmap + json + parallel feature
cargo build

# 发布构建
cargo build --release

# 构建 GUI（egui 桌面界面）
cargo build --release --features gui

# 运行 GUI
cargo run --release --features gui --bin gguf-gui

# 无默认 features（纯解析，无 mmap/json/parallel）
cargo build --no-default-features

# 运行全部测试
cargo test

# 质量检查
cargo clippy --all-features --all-targets
cargo fmt --check
```

### Cargo Features

| Feature | 默认 | 说明 |
|---------|------|------|
| `mmap` | ✅ | `from_path` 优先内存映射（依赖 C 工具链） |
| `json` | ✅ | `gguf-dump --json` 输出 + `value_to_json` 库 API |
| `parallel` | ✅ | rayon 并行化反量化 / matvec |
| `gui` | — | egui/eframe/rfd 桌面 GUI（`gguf-gui` bin） |

> Windows 下 `mmap` feature 依赖 C 工具链（MSVC / GNU）。若无可用工具链：
>
> ```bash
> cargo build --no-default-features --features json
> ```

## 使用 CLI（gguf-dump）

```bash
gguf-dump [OPTIONS] <PATH>
```

### 参数

| 参数 | 说明 |
|------|------|
| `<PATH>` | GGUF 文件路径 |
| `-j, --json` | 以 JSON 格式输出（默认文本） |
| `--pretty` | JSON 美化输出（仅对 `--json` 有意义） |
| `-t, --tensors-all` | 文本模式下显示全部张量（默认截断前 50） |
| `-m, --max-kv <N>` | 文本模式下 KV 显示上限（默认 200） |
| `-k, --key <KEY>` | 仅显示指定键的 KV 值（可多次指定） |
| `--summary-only` | 仅显示文件摘要 |
| `-q, --quiet` | 静默模式：仅摘要，不显示张量与 KV |
| `-T, --token-id <ID>` | 按 token id 查词表（可多次） |
| `-S, --token-str <STR>` | 按 token 字符串查词表（可多次） |
| `--merge-contains <SUB>` | 打印 BPE merges 中匹配子串的条目（可多次） |
| `-h, --help` | 帮助 |
| `-V, --version` | 版本 |

### 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 文件不存在 / 无法打开（I/O 错误） |
| 2 | 非 GGUF 文件（魔数错误） |
| 3 | 版本不支持 |
| 4 | 文件损坏 / 解析错误 |

### 示例

```bash
# 默认文本输出（摘要 + KV + 前 50 张量）
gguf-dump model.gguf

# 仅查看摘要
gguf-dump --summary-only model.gguf

# 仅看某个键
gguf-dump -k general.architecture -k llama.block_count model.gguf

# JSON 输出（可管道给 jq / 反序列化）
gguf-dump --json model.gguf

# JSON 美化
gguf-dump --json --pretty model.gguf
```

### 文本输出样例

```
GGUF File: gist-embedding-v0.Q2_K.gguf
=====================
Size:            51.72 MB
Version:         3
Tensors:         197
KV pairs:        24
Alignment:       32
Data offset:     760,992 bytes (743.16 KB)
Architecture:    bert
Model name:      GIST-Embedding-v0

Key-Value Metadata (showing 24 of 24):
  general.architecture                   string "bert"
  general.name                           string "GIST-Embedding-v0"
  bert.block_count                       uint32 U32(12)
  bert.context_length                    uint32 U32(512)
  tokenizer.ggml.tokens                  array [30522 elements] (first 5: "[PAD]", "[unused0]", ...)
  ...

Tensors (showing first 50 of 197):
  NAME                                     SHAPE                    TYPE                     OFFSET               SIZE
  token_embd_norm.bias                     [768]                    F32                0              3,072
  position_embd.weight                     [768, 512]               F32            6,144          1,572,864
  token_embd.weight                        [768, 30522]             Q4_K        1,585,152                  —
  ...
```

### JSON 输出样例

```json
{
  "file": "model.gguf",
  "size": 54236512,
  "header": { "magic": "46554747", "version": 3, "tensors": 197, "kv_pairs": 24 },
  "alignment": 32,
  "data_offset": 760992,
  "architecture": "bert",
  "model_name": "GIST-Embedding-v0",
  "kv": {
    "general.architecture": { "type": "string", "value": "bert" },
    "bert.block_count": { "type": "uint32", "value": 12 }
  },
  "tensors": [
    { "name": "token_embd_norm.bias", "shape": [768], "dtype": "F32",
      "offset": 0, "num_elements": 768 }
  ]
}
```

> JSON 中数组值若元素数超过 1000，`value` 仅放前 1000 项并附 `"truncated": true, "total": <N>`，避免输出爆炸。

## 使用 CLI（gguf-infer）

```bash
gguf-infer [OPTIONS] <PATH>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<PATH>` | GGUF 模型文件路径 | — |
| `-p, --prompt <TEXT>` | 输入 prompt（缺省从 stdin 读取） | — |
| `-n, --max-tokens <N>` | 最大生成 token 数 | 128 |
| `-t, --temperature <F>` | 温度（0 = 贪心） | 0.8 |
| `--top-k <N>` | Top-K（0 = 禁用） | 40 |
| `--top-p <F>` | Top-P 阈值（1.0 = 禁用） | 0.95 |
| `--min-p <F>` | Min-P 相对概率阈值（0.0 = 禁用） | 0.0 |
| `--repeat-penalty <F>` | 重复惩罚（1.0 = 禁用） | 1.1 |
| `-s, --seed <N>` | 随机种子（0 = 系统随机） | 0 |
| `--greedy` | 强制贪心解码（忽略 temperature/top-k/top-p） | false |
| `-v, --verbose` | 打印统计信息（耗时等） | false |
| `--no-stream` | 禁用流式输出（一次性打印） | stream 默认开启 |
| `--chat` | 交互问答模式（`:reset` 重置 / `:quit` 退出） | false |
| `-h, --help` | 帮助 | — |
| `-V, --version` | 版本 | — |

### 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 文件不存在 / 无法打开（I/O 错误） |
| 2 | 非 GGUF 文件（魔数错误） |
| 3 | 版本不支持 |
| 4 | 架构不支持 |
| 5 | 分词器错误 |
| 6 | 缺少张量 |
| 7 | 其他推理错误 |

### 示例

```bash
# 基本用法（流式输出，默认参数）
gguf-infer -p "你好" model.gguf

# 贪心解码 + 512 tokens
gguf-infer model.gguf -p "Explain quantum computing" --greedy -n 512

# 自定义采样参数
gguf-infer model.gguf -p "写一首诗" -t 0.7 --top-k 20 --top-p 0.9 --min-p 0.05

# 从 stdin 读取 prompt
echo "Tell me a joke" | gguf-infer model.gguf

# 非流式 + 详细输出
gguf-infer model.gguf -p "Hi" --no-stream -v

# 交互问答模式（上下文持续累积）
gguf-infer model.gguf --chat
```

## 使用 GUI（gguf-gui）

```bash
# 构建并运行（需启用 gui feature）
cargo run --release --features gui --bin gguf-gui
```

基于 **egui/eframe** 的桌面 GUI，三大标签页：

| 标签页 | 功能 |
|--------|------|
| **对话** | 多轮对话，Qwen2 chat template，流式逐 token 显示，支持停止生成 / 重置上下文 / 上下文超限提示 |
| **Prompt 补全** | 单轮 prompt 输入，支持流式/非流式、贪心/采样，显示 tok/s / 耗时 / ctx 使用 |
| **模型信息** | 文件摘要、KV 元数据（可搜索）、张量列表（默认前 50）、词表查询（id↔string）、JSON 导出 |

**采样参数面板**（顶部工具栏，可折叠）：温度 / Top-K / Top-P / Min-P / 重复惩罚 / 随机种子 / 最大 token / 贪心开关。

**架构设计**：UI 线程通过 mpsc 通道与后台推理线程通信，`Engine` 仅在推理线程中持有；`stop_flag`（`Arc<AtomicBool>`）实现生成中断；`catch_unwind` 防止推理 panic 导致线程静默死亡。

## 使用库 API（gguf crate）

在你的项目 `Cargo.toml` 中添加：

```toml
[dependencies]
gguf = { path = "..." }            # 或发布后写版本
# 仅用 mmap： gguf = { version = "0.1", features = ["mmap"] }
```

### 快速开始

```rust
use gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 从文件路径解析（优先 mmap）
    let file = GgufFile::from_path("model.gguf")?;

    println!("architecture = {:?}", file.architecture());
    println!("model name   = {:?}", file.model_name());
    println!("tensors      = {}", file.header.n_tensors);
    println!("kv pairs     = {}", file.header.n_kv);
    println!("data offset  = {}", file.data_offset);
    Ok(())
}
```

### 三种读取入口

```rust
// 1. 内存缓冲（零拷贝读，缓冲生命周期需覆盖 GgufFile 使用期时自行克隆）
let file = GgufFile::from_buffer(&data)?;

// 2. 文件路径（优先 mmap，失败回退整体读取；只读元数据区，不加载权重数据体）
let file = GgufFile::from_path("model.gguf")?;

// 3. 任意 Read + Seek 读取器（整体读入内存）
let file = GgufFile::from_reader(std::io::Cursor::new(data))?;
```

### 常用方法

```rust
// 按键查找 KV 值（未命中返回 None）
if let Some(v) = file.get("llama.block_count") {
    println!("block_count = {:?}", v.as_i64());
}

// 按名查找张量
if let Some(t) = file.find_tensor("token_embd.weight") {
    println!("shape = {:?}, dtype = {}, elements = {}",
             t.shape, t.dtype, t.num_elements());
}

// KV 转 HashMap，便于批量查询 / 序列化
let map = file.kv_map();
```

### 核心类型

| 类型 | 说明 |
|------|------|
| `GgufFile` | 解析结果：`header` / `kv` / `tensors` / `alignment` / `data_offset` / `file_size` |
| `GgufHeader` | 文件头：`magic` / `version` / `n_tensors` / `n_kv` |
| `GgufValue` | KV 值动态枚举（13 种标量 + `Array`），提供 `as_*` 与 `display()` |
| `GgufArray` | 数组：`elem_type` + `data`（元素同质，不允许嵌套） |
| `TensorInfo` | 张量描述符：`name` / `shape` / `dtype` / `offset`，提供 `num_elements()` / `est_data_size()` |
| `GgmlType` | 张量数据类型（常见量化值具名，未知值 `Unknown(i32)`） |
| `GgufError` | 错误枚举（I/O / 魔数 / 版本 / 越界 / 非法类型 / 推理错误 等） |

### 推理 API

```rust
use gguf::GgufFile;
use gguf::infer::Engine;
use gguf::infer::sampler::SamplerConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::from_path("model.gguf")?;

    // 构建引擎（贪心解码）
    let cfg = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        seed: 42,
    };
    let mut engine = Engine::new(&file, cfg)?;

    // 分词 / 反分词
    let ids = engine.tokenize("hello world");
    let text = engine.detokenize(&ids);

    // 一次性补全
    let result = engine.complete("Tell me a joke.", 64)?;
    println!("{result}");

    // 流式生成（逐 token 回调）
    engine.generate("Write a poem about", 128, |token_id, text| {
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    })?;

    // 多轮对话（Qwen2 chat template）
    let reply = engine.chat("你好，请介绍一下自己", 256)?;
    println!("{reply}");

    // 重置对话上下文
    engine.reset();

    // 查看模型超参
    let hp = engine.hparams();
    println!("embed={}, heads={}, kv_heads={}, ffn={}, ctx={}",
             hp.embed_dim, hp.n_heads, hp.n_kv_heads, hp.ffn_dim, hp.context_length);

    Ok(())
}
```

| 类型 / 方法 | 说明 |
|-------------|------|
| `Engine::new(&file, cfg)` | 从 GGUF 文件构建推理引擎 |
| `Engine::generate(prompt, max, on_token)` | 逐 token 流式生成（回调 `(token_id, &str)`） |
| `Engine::complete(prompt, max)` | 一次性补全，返回完整文本 |
| `Engine::chat(text, max)` | 多轮对话（Qwen2 chat template），返回回复文本 |
| `Engine::reset()` | 重置对话上下文（清空 cache + history） |
| `Engine::tokenize(text)` | 文本 → token id 列表 |
| `Engine::detokenize(&ids)` | token id 列表 → 文本 |
| `Engine::hparams()` | 模型超参数（`embed_dim` / `n_heads` / `n_kv_heads` / `ffn_dim` / `context_length`） |
| `SamplerConfig` | 采样配置：`temperature` / `top_k` / `top_p` / `min_p` / `repeat_penalty` / `seed` |

## 设计要点

- **零权重加载**：解析层只读取 header + KV + 张量描述符（文件前部），数据体不加载 → 解析数十 GB 模型也很快
- **mmap 优先**：`from_path` 先尝试内存映射，内核按需换页；元数据区集中在文件前部，实际物理读取量小
- **边界保护**：每次读取前校验剩余字节，越界返回 `OutOfBounds{offset, required, file_size}`
- **大数组防 OOM**：解析前用 `count * elem_min_size ≤ remaining` 预检，损坏的超大 count 直接报错
- **优雅降级**：未知 `ggml_type` 显示为 `Unknown(n)`，不中断解析
- **KV Cache 增量推理**：prompt 预填充一次 forward，后续逐 token 复用历史 KV，避免重复计算
- **量化 matvec**：Q4_0~Q8_K 权重在反量化时直接参与矩阵向量乘，无需完整反量化到 F32
- **生产代码无 `unwrap` / `expect` / `panic!`**：所有错误路径返回 `Err`

## 测试

```bash
cargo test
```

**190 个测试全部通过**（0 failed）

- 单元测试（137）：`Cursor` 各类型读取 / 越界、`GgufValue::display()`、`TensorInfo` 方法、`GgufType` / `GgmlType` 映射、量化反量化 roundtrip（Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / K-quant）、量化 matvec 与 F32 对照、采样器（贪心 / 温度 / Top-K / Top-P / Min-P / Repeat Penalty）、分词器 encode/decode
- 集成测试 `tests/parse_buffer.rs`（16）：内存构造合法 GGUF 缓冲，覆盖全部 13 种 KV 类型、字符串、负数、f64、数组、多张量、未知 dtype
- 集成测试 `tests/inference.rs`（5）：`gguf-infer` 端到端推理（构造完整 llama GGUF → 加载 → 生成）
- CLI 测试 `tests/cli.rs`（21）：`gguf-dump` 文本输出关键键、`--json` 可反序列化、退出码（1/2/4）、大数组截断
- 健壮性测试：错误 magic / version、任意截断、损坏数组 count、嵌套数组、空 / 短文件

## Windows 中文支持

- **CLI**：`gguf-infer` 交互模式通过 `ReadConsoleW`/`WriteConsoleW` 宽字符 API 读写，管道场景回退字节级 UTF-8；真实终端自动切换 CP 65001
- **GUI**：加载 `msyh.ttc` 中文字体，插入 Proportional 字体族，解决 egui 默认字体不支持中文的问题

## 许可

MIT
