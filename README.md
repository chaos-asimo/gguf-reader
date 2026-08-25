# gguf — GGUF 元数据读取程序

一个纯 Rust 实现，用于读取大模型二进制存储格式 **GGUF**（GGML Universal Format，llama.cpp 生态）文件的**元数据**：文件头、KV 键值元数据、张量描述符。**不读取**张量权重数据体，因此可高效处理数十 GB 的模型文件。

同时提供：

- **`gguf` 库 crate**：可复用的解析 API
- **`gguf-dump` CLI 工具**：查看任意 GGUF 文件的元数据，支持文本与 JSON 输出

## 特性

- 完整解析 GGUF version 3 文件头 + 全部 **13 种** `gguf_type`（含数组）
- 解析张量描述符（名称 / 形状 / 数据类型 / 数据偏移 / 元素数 / 估算大小）
- **零权重加载**：仅读取元数据区（通常 < 几十 MB），适合大文件
- `from_path` 优先 **mmap** 零拷贝，失败自动回退普通读取
- 健壮的错误处理：损坏 / 截断 / 非法 UTF-8 / 大数组 count 均返回 `Err`，**不 panic**
- 未知张量 dtype 优雅降级为 `GgmlType::Unknown(n)`
- CLI 支持文本摘要、KV 列表、张量表格，以及 `--json`（大数组自动截断）

## 项目结构

```
model_study/
├── Cargo.toml              # 包定义（features: mmap / json）
├── src/
│   ├── lib.rs              # gguf 库入口，re-export 公共类型
│   ├── types.rs            # GgufType, GgmlType, GgufValue, GgufArray
│   ├── tensor.rs           # TensorInfo
│   ├── header.rs           # GgufHeader
│   ├── file.rs             # GgufFile（核心解析）
│   ├── cursor.rs           # Cursor 字节读取器（内部，带边界保护）
│   ├── error.rs            # GgufError
│   └── bin/
│       └── gguf_dump.rs    # CLI 入口
├── tests/
│   ├── parse_buffer.rs     # 基于内存缓冲的解析测试（覆盖 13 种 KV 类型）
│   ├── cli.rs              # CLI 集成测试
│   └── common/             # 测试工具（构造内存 GGUF 缓冲）
├── spec.md                 # 技术规范
├── checklist.md            # 验收清单
└── tasks.md                # 任务拆解
```

## 构建

```bash
# 默认开启 mmap 与 json feature
cargo build

# 发布构建
cargo build --release

# 运行全部测试
cargo test

# 质量检查
cargo clippy --all-targets
cargo fmt --check
```

> Windows 下 `mmap` feature 依赖 C 工具链（MSVC / GNU）。若无可用工具链，可关闭该 feature 回退到普通读取：
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
| `--tensors-all` | 文本模式下显示全部张量（默认截断前 50） |
| `-m, --max-kv <N>` | 文本模式下 KV 显示上限（默认 200） |
| `-k, --key <KEY>` | 仅显示指定键的 KV 值（可多次指定） |
| `--summary-only` | 仅显示文件摘要 |
| `-q, --quiet` | 静默模式：仅摘要，不显示张量与 KV |
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
| `GgufError` | 错误枚举（I/O / 魔数 / 版本 / 越界 / 非法类型 等） |

## 设计要点

- **零权重加载**：只读取 header + KV + 张量描述符（文件前部），数据体不加载 → 解析数十 GB 模型也很快
- **mmap 优先**：`from_path` 先尝试内存映射，内核按需换页；元数据区集中在文件前部，实际物理读取量小
- **边界保护**：每次读取前校验剩余字节，越界返回 `OutOfBounds{offset, required, file_size}`
- **大数组防 OOM**：解析前用 `count * elem_min_size ≤ remaining` 预检，损坏的超大 count 直接报错
- **优雅降级**：未知 `ggml_type` 显示为 `Unknown(n)`，不中断解析
- **生产代码无 `unwrap` / `expect` / `panic!`**：所有错误路径返回 `Err`

## 测试

```bash
cargo test
```

- 单元测试：`Cursor` 各类型读取 / 越界、`GgufValue::display()`、`TensorInfo::num_elements` / `est_data_size`、`GgufType` / `GgmlType` 映射
- 集成测试（`tests/parse_buffer.rs`）：内存构造合法 GGUF 缓冲，覆盖全部 13 种 KV 类型、字符串、负数、f64、数组、多张量、未知 dtype
- 健壮性测试：错误 magic / version、任意截断、损坏数组 count、嵌套数组、空 / 短文件
- CLI 测试（`tests/cli.rs`）：文本输出关键键、`--json` 可反序列化、退出码（1/2/4）、大数组截断

## 许可

MIT
