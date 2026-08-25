# GGUF 元数据读取程序 — 技术规范 (Specification)

## 1. 项目概述

### 1.1 背景
GGUF（GGML Universal Format）是 llama.cpp 生态用于存储大语言模型权重及元数据的二进制文件格式。它把模型权重、分词器信息、架构超参数等全部封装进单个自包含文件，支持内存映射（mmap）零拷贝加载。

### 1.2 目标
开发一个 **Rust** 程序，能够：
- 完整解析 GGUF 文件的 **元数据**（文件头 + KV 键值元数据 + 张量描述符）
- **不读取**张量权重数据本身（数据体可能达数十 GB，仅解析其描述信息）
- 提供可复用的 **库 API**（`gguf` crate）
- 提供 **CLI 工具**（`gguf-dump`）用于查看任意 GGUF 文件的元数据，支持文本与 JSON 输出

### 1.3 非目标（Out of Scope）
- 不实现 GGUF 写入功能
- 不解析张量权重二进制数据体（仅解析偏移量/形状等描述）
- 不实现模型推理
- 不依赖 llama.cpp 的 C 库，纯 Rust 实现

---

## 2. GGUF 格式规范（权威参考）

来源：llama.cpp 官方 `ggml/include/gguf.h`。

### 2.1 文件整体结构（按字节顺序）

```
┌─────────────────────────────────────────────┐
│ 1. Header                                   │
│    magic      : 4 bytes  "GGUF" (0x46554747)│
│    version    : uint32   (当前 = 3)         │
│    n_tensors  : int64    张量数量            │
│    n_kv       : int64    键值对数量          │
├─────────────────────────────────────────────┤
│ 2. KV Metadata (n_kv 个键值对)              │
│    对每个 KV：                               │
│      key  : string (uint64 长度 + UTF-8 字节)│
│      type : gguf_type (int32)               │
│      value: 取决于 type：                    │
│        - 若 GGUF_TYPE_ARRAY：               │
│            elem_type : gguf_type (int32)     │
│            count     : uint64                │
│            elements  : count 个元素(按 elem) │
│        - 否则：                              │
│            单值二进制表示                     │
├─────────────────────────────────────────────┤
│ 3. Tensor Info (n_tensors 个张量)           │
│    对每个张量：                              │
│      name   : string                        │
│      n_dims : uint32                        │
│      ne[]   : n_dims 个 int64（维度大小，    │
│               存储顺序为 ne[n_dims-1-i]）    │
│      dtype  : ggml_type (int32)             │
│      offset : uint64（数据体中的偏移）        │
├─────────────────────────────────────────────┤
│ 4. (对齐填充 pad)                           │
│    填充到 alignment 边界                     │
├─────────────────────────────────────────────┤
│ 5. Tensor Data Blob (可选，本程序不读取)     │
└─────────────────────────────────────────────┘
```

### 2.2 序列化规则
- **字符串**：`uint64` 长度前缀 + UTF-8 字节（**不含** null 终止符）
- **所有枚举**：存为 `int32`
- **布尔值**：存为 `int8`
- **对齐**：若 KV 中存在键 `general.alignment`（uint32），用其值作为数据体对齐；否则默认 `GGUF_DEFAULT_ALIGNMENT = 32`
- **字节序**：默认小端（Little-Endian）。读取时按主机字节序，本程序仅支持小端（GGUF 实际文件均为小端）

### 2.3 KV 值类型枚举 `gguf_type`

| 值  | 名称          | 二进制表示 | 大小     |
|-----|---------------|-----------|----------|
| 0   | UINT8         | u8        | 1 B      |
| 1   | INT8          | i8        | 1 B      |
| 2   | UINT16        | u16       | 2 B      |
| 3   | INT16         | i16       | 2 B      |
| 4   | UINT32        | u32       | 4 B      |
| 5   | INT32         | i32       | 4 B      |
| 6   | FLOAT32       | f32       | 4 B      |
| 7   | BOOL          | i8 (0/1)  | 1 B      |
| 8   | STRING        | 长度+字节 | 变长     |
| 9   | ARRAY         | 见 2.1    | 变长     |
| 10  | UINT64        | u64       | 8 B      |
| 11  | INT64         | i64       | 8 B      |
| 12  | FLOAT64       | f64       | 8 B      |

### 2.4 张量数据类型 `ggml_type`（部分常见值）

| 值  | 名称  | 值  | 名称    | 值  | 名称   |
|-----|-------|-----|---------|-----|--------|
| 0   | F32   | 1   | F16     | 2   | Q4_0   |
| 3   | Q4_1  | 4   | Q4_2    | 5   | Q4_3   |
| 6   | Q5_0  | 7   | Q5_1    | 8   | Q8_0   |
| 9   | Q8_1  | 10  | Q2_K    | 11  | Q3_K_L |
| 12  | Q3_K_M| 13  | Q3_K_S  | 14  | Q4_K   |
| 15  | Q5_K  | 16  | Q6_K    | 17  | Q8_K   |
| 18  | IQ2_XXS| 19 | IQ2_XS  | 20  | IQ3_XXS|
| 21  | IQ1_S | 22  | IQ4_NL  | 23  | IQ3_S  |
| 24  | IQ2_S | 25  | IQ2_M   | 26  | IQ3_M  |
| 27  | IQ1_M | 28  | IQ4_XS  | 29  | IQ3_XS |
| 30  | BF16  | 31  | Q2_K    | ... | ...    |

> 程序对未知 dtype 值应优雅降级（显示为 `UnknownType(<n>)`），而非崩溃。

### 2.5 常见元数据键（示例，用于理解语义，非强制解析）

| 键 | 说明 |
|----|------|
| `general.architecture` | 模型架构名（如 `llama`、`qwen2`、`gemma`）|
| `general.name` | 模型名称 |
| `general.file_type` | 文件类型 |
| `general.alignment` | 张量数据对齐（uint32）|
| `general.quantization_version` | 量化版本 |
| `{arch}.vocab_size` | 词表大小 |
| `{arch}.context_length` | 上下文长度 |
| `{arch}.block_count` | 块（层）数量 |
| `{arch}.embedding_length` | 嵌入维度 |
| `tokenizer.ggml.model` | 分词器类型 |
| `tokenizer.ggml.tokens` | 分词器 token 数组（string[]）|
| `tokenizer.ggml.scores` | 分词器分数数组（f32[]）|
| `split.no` / `split.count` | 分片编号 / 分片总数 |

> 库需能解析**任意**键名与类型，无需硬编码上述键；它们是语义参考。

---

## 3. 数据模型（Rust 类型设计）

### 3.1 枚举

```rust
/// GGUF KV 元数据类型（对应 gguf_type）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum GgufType {
    Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
    Uint32 = 4, Int32 = 5, F32 = 6, Bool = 7,
    String = 8, Array = 9, Uint64 = 10, Int64 = 11, F64 = 12,
}

/// 张量数据类型（对应 ggml_type），常见值具名，未知值用 Unknown
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum GgmlType {
    F32 = 0, F16 = 1, Q4_0 = 2, /* ... 常见量化类型 ... */ BF16 = 30,
    Unknown(i32),
}
```

### 3.2 KV 值（动态类型）

```rust
/// 单个 KV 元数据的值
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// 数组：元素类型 + 元素序列
    Array(GgufArray),
}

/// 数组类型，元素必须同质
#[derive(Clone, Debug, PartialEq)]
pub struct GgufArray {
    pub elem_type: GgufType,   // 不含 Array（GGUF 不允许嵌套数组）
    pub data: Vec<GgufValue>,  // 每个元素为标量 GgufValue
}

impl GgufValue {
    /// 返回值的类型标签
    pub fn value_type(&self) -> GgufType { ... }
    /// 便捷提取（类型不匹配时返回 None）
    pub fn as_str(&self) -> Option<&str> { ... }
    pub fn as_i64(&self) -> Option<i64> { ... }
    pub fn as_f64(&self) -> Option<f64> { ... }
    pub fn as_bool(&self) -> Option<bool> { ... }
    pub fn as_array(&self) -> Option<&GgufArray> { ... }
    /// 人类可读字符串表示（用于 CLI 文本输出）
    pub fn display(&self) -> String { ... }
}
```

### 3.3 张量描述符

```rust
/// 单个张量的元数据描述（不含权重数据）
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// 各维度大小；存储顺序与文件一致（已按 ne[n_dims-1-i] 还原为逻辑顺序）
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// 张量数据在数据体中的字节偏移
    pub offset: u64,
}

impl TensorInfo {
    /// 元素总数 = shape 各维之积
    pub fn num_elements(&self) -> u64 { ... }
    /// 按 dtype 估算单个元素字节数（量化类型按 block 估算，保守返回下界）
    pub fn est_element_size(&self) -> Option<u64> { ... }
    /// 估算张量数据字节数
    pub fn est_data_size(&self) -> Option<u64> { ... }
}
```

### 3.4 文件头

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgufHeader {
    pub magic: u32,        // 应等于 0x46554747
    pub version: u32,      // 当前为 3
    pub n_tensors: u64,    // 由 int64 读入，取非负
    pub n_kv: u64,
}
```

### 3.5 顶层结构

```rust
/// 解析后的 GGUF 文件元数据（不含权重数据）
#[derive(Clone, Debug)]
pub struct GgufFile {
    pub header: GgufHeader,
    /// KV 元数据，保持文件内出现顺序
    pub kv: Vec<(String, GgufValue)>,
    /// 张量描述符，保持文件内出现顺序
    pub tensors: Vec<TensorInfo>,
    /// 解析出的对齐值（来自 general.alignment，缺省 32）
    pub alignment: u32,
    /// 元数据区结束、数据体起始的文件偏移（含对齐填充）
    pub data_offset: u64,
    /// 文件总字节数（用于校验 data_offset 不越界）
    pub file_size: u64,
}

impl GgufFile {
    /// 按键查找 KV 值
    pub fn get(&self, key: &str) -> Option<&GgufValue> { ... }
    /// 按名查找张量
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> { ... }
    /// 架构名（general.architecture）便捷访问
    pub fn architecture(&self) -> Option<&str> { ... }
    /// 模型名（general.name）便捷访问
    pub fn model_name(&self) -> Option<&str> { ... }
    /// KV 转 HashMap（便于 JSON 序列化与快速查询）
    pub fn kv_map(&self) -> HashMap<String, &GgufValue> { ... }
}
```

---

## 4. API 设计（`gguf` crate 公共接口）

### 4.1 读取入口

```rust
impl GgufFile {
    /// 从内存缓冲解析（零拷贝读，缓冲需保持存活于 GgufFile 之外时克隆）
    pub fn from_buffer(data: &[u8]) -> Result<Self, GgufError>;

    /// 从文件路径解析：优先 mmap，失败回退到整体读入内存
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GgufError>;

    /// 从任意 Read+Seek 读取器解析（整体读入内存）
    pub fn from_reader<R: Read + Seek>(reader: R) -> Result<Self, GgufError>;
}
```

> 设计说明：`GgufFile` 内部持有 `Vec<u8>`（mmap 时拷贝元数据区，或 mmap 视图 + 元数据区克隆）。为简化所有权与跨 FFI，统一在解析时把 **元数据区**（header+kv+tensor info）拷贝为 owned `Vec<u8>`，权重数据体不加载。mmap 优势体现在：只把元数据区（通常 < 几十 MB）读入，避免一次性读入整个数十 GB 文件。

### 4.2 错误类型

```rust
#[derive(Debug)]
pub enum GgufError {
    /// 文件 I/O 错误
    Io(std::io::Error),
    /// mmap 错误
    Mmap(String),
    /// 魔数不匹配（非 GGUF 文件）
    InvalidMagic(u32),
    /// 不支持的版本
    UnsupportedVersion(u32),
    /// 读取越界（文件损坏/截断）
    OutOfBounds { offset: u64, required: u64, file_size: u64 },
    /// 非法的 KV 类型值
    InvalidGgufType(i32),
    /// 非法的张量 dtype
    InvalidGgmlType(i32),
    /// 数组元素类型非法（如嵌套数组）
    InvalidArrayElemType(i32),
    /// 字符串长度非法（如超过剩余字节）
    InvalidStringLength(u64),
    /// 张量维度非法（负数等）
    InvalidTensorDim { name: String, dim: i64 },
    /// KV 键名重复（记录但可容忍，依实现策略）
    DuplicateKey(String),
    /// 其他
    Other(String),
}
impl fmt::Display for GgufError { ... }
impl std::error::Error for GgufError { ... }
impl From<std::io::Error> for GgufError { ... }
```

### 4.3 字节读取辅助（内部）

```rust
/// 带边界保护的序读取器
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Cursor<'_> {
    fn u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/bool(&mut self) -> Result<T, GgufError>;
    fn string(&mut self) -> Result<String, GgufError>;
    fn remaining(&self) -> usize;
}
```
- 所有读取前校验剩余字节 ≥ 所需，不足返回 `OutOfBounds`
- 字节序：小端（`u16::from_le_bytes` 等）

---

## 5. CLI 工具设计（`gguf-dump`）

### 5.1 命令行接口（clap derive）

```
gguf-dump [OPTIONS] <PATH>

位置参数:
  <PATH>               GGUF 文件路径

选项:
  -j, --json           以 JSON 格式输出（默认文本）
      --pretty         JSON 美化输出（仅对 --json 有意义）
  -t, --tensors        显示张量列表（默认在文本模式显示摘要）
      --tensors-all    文本模式下显示全部张量（默认截断至前 50 条）
  -m, --max-kv <N>     文本模式下 KV 显示上限（默认 200，数组大值截断显示）
  -k, --key <KEY>      仅显示指定键的 KV 值（可多次）
      --summary-only   仅显示文件摘要（header + 对齐 + 数量统计）
  -q, --quiet          不显示张量与 KV，仅摘要
  -h, --help           帮助
  -V, --version        版本
```

### 5.2 退出码
| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 文件不存在 / 无法打开（I/O 错误）|
| 2 | 非 GGUF 文件（魔数错误）|
| 3 | 版本不支持 |
| 4 | 文件损坏 / 解析错误 |

### 5.3 文本输出样例（默认）

```
GGUF File: model.gguf
=====================
Size:            4.87 GB
Version:         3
Tensors:         1,532
KV pairs:        87
Alignment:       32
Data offset:     1,048,576 bytes
Architecture:    llama
Model name:      Llama-3-8B-Instruct

Key-Value Metadata (showing 87):
  general.architecture       string  "llama"
  general.name               string  "Llama-3-8B-Instruct"
  general.alignment          uint32  32
  llama.block_count          uint32  32
  llama.embedding_length     uint32  4096
  llama.vocab_size           uint32  128256
  ...
  tokenizer.ggml.tokens      array<string>  [32000 elements] (first 10: <s>, </s>, ..., ...)

Tensors (showing first 50 of 1532):
  NAME                                    SHAPE               TYPE    OFFSET      SIZE
  token_embd.weight                       [128256, 4096]      BF16    1048576     1,050,657,280
  output.norm.weight                      [4096]              F32     ...         16,384
  blk.0.attn_q.weight                     [4096, 4096]        Q8_0    ...         16,777,280
  ...
```

### 5.4 JSON 输出样例

```json
{
  "file": "model.gguf",
  "size": 5242880000,
  "header": { "version": 3, "tensors": 1532, "kv_pairs": 87 },
  "alignment": 32,
  "data_offset": 1048576,
  "kv": {
    "general.architecture": { "type": "string", "value": "llama" },
    "general.name": { "type": "string", "value": "Llama-3-8B-Instruct" },
    "llama.block_count": { "type": "uint32", "value": 32 },
    "tokenizer.ggml.tokens": {
      "type": "array", "element_type": "string", "count": 32000,
      "value": ["<s>", "</s>", "..."]
    }
  },
  "tensors": [
    { "name": "token_embd.weight", "shape": [128256, 4096],
      "dtype": "BF16", "offset": 1048576, "num_elements": 525316576 }
  ]
}
```

> JSON 中数组值若元素数超过阈值（默认 1000），`value` 仅放前 1000 项并附 `"truncated": true, "total": 32000`，避免输出爆炸。

---

## 6. 项目结构

```
model_study/
├── Cargo.toml              # workspace + 包定义
├── src/
│   ├── lib.rs              # gguf 库入口，re-export 公共类型
│   ├── types.rs            # GgufType, GgmlType, GgufValue, GgufArray
│   ├── tensor.rs           # TensorInfo
│   ├── header.rs           # GgufHeader
│   ├── file.rs             # GgufFile
│   ├── cursor.rs           # Cursor 字节读取器（内部）
│   ├── error.rs            # GgufError
│   └── bin/
│       └── gguf_dump.rs    # CLI 入口
├── tests/
│   ├── parse_buffer.rs     # 基于内存缓冲的解析测试
│   ├── fixtures/           # 测试用小型 GGUF 文件（构造生成）
│   └── cli.rs              # CLI 集成测试
├── spec.md
├── checklist.md
└── tasks.md
```

### 6.1 依赖
| Crate | 版本 | 用途 |
|-------|------|------|
| `clap` | 4.x | CLI 解析（derive 特性）|
| `serde` | 1.x | 序列化（derive 特性）|
| `serde_json` | 1.x | JSON 输出 |
| `memmap2` | 0.9.x | mmap 读取（可选特性 `mmap`）|
| `anyhow` | 1.x | CLI 层错误处理 |
| `libc` / `winapi` | — | mmap 跨平台（memmap2 已封装，通常无需直接依赖）|

> 库本身尽量零依赖（`memmap2` 作为可选 feature，默认开启）；CLI 二进制引入 clap/serde/anyhow。

---

## 7. 错误处理与边界策略

1. **魔数校验**：首 4 字节必须为 `0x46554747`（"GGUF"），否则 `InvalidMagic`
2. **版本检查**：version 必须等于 3（或可配置的 ≤3），否则 `UnsupportedVersion`
3. **边界保护**：每次读取前校验剩余字节，防止越界（`OutOfBounds`），抵御截断/损坏文件
4. **字符串安全**：长度前缀不得超出剩余字节；UTF-8 解码失败返回 `InvalidStringLength` 或 `Other`
5. **数组防嵌套**：数组元素类型若为 `GGUF_TYPE_ARRAY` 则报 `InvalidArrayElemType`
6. **未知枚举值**：`GgmlType::Unknown(n)` 优雅降级；`GgufType` 未知值报错（因无法解析）
7. **负数计数**：`n_kv` / `n_tensors` 读入为 int64，若为负或超过剩余容量则报错
8. **大数组保护**：数组 count 若导致所需字节远超剩余（如损坏文件），在解析前用 `count * elem_size` 预估并校验，避免 OOM
9. **重复键**：记录警告但仍保留（依实现，默认覆盖或追加；本规范选追加并标记 `DuplicateKey` 可选返回）
10. **mmap 失败回退**：mmap 出错时回退到 `read` 整体读入元数据区，不致命

---

## 8. 性能考量

- **零权重加载**：只读取 header+kv+tensor info（通常 < 几十 MB），数据体不加载 → 适合数十 GB 模型
- **mmap**：`from_path` 优先 mmap，内核按需换页；元数据区集中在文件前部，实际物理读取量小
- **解析复杂度**：O(n_kv * 平均值大小 + n_tensors)，线性
- **JSON 大数组截断**：避免序列化 32000 个 token 字符串导致输出/内存爆炸

---

## 9. 测试策略

### 9.1 单元测试（`types` / `cursor` / `tensor`）
- `GgufValue::display()` 各类型正确性
- `Cursor` 越界检测
- `TensorInfo::num_elements` / `est_data_size`
- `GgufType::from_i32` / `GgmlType::from_i32` 映射

### 9.2 集成测试（构造缓冲）
- 用测试工具函数在内存中构造合法 GGUF 缓冲（header + 若干 KV + 若干 tensor info + 假数据体），断言解析结果
- 覆盖全部 13 种 KV 类型 + 数组
- 覆盖字符串、负数 int、f64 等
- 覆盖多张量、0 维/高维 shape

### 9.3 健壮性测试
- 魔数错误 → `InvalidMagic`
- 版本错误 → `UnsupportedVersion`
- 截断文件（任意位置截断）→ `OutOfBounds` / 合适错误
- 损坏数组 count → 不 OOM、报错
- 嵌套数组 → `InvalidArrayElemType`
- 空文件 / 短文件

### 9.4 CLI 测试
- 文本输出包含关键键
- `--json` 输出可被 `serde_json` 反序列化且字段正确
- 退出码：不存在文件=1、非 GGUF=2、损坏=4
- 大数组截断生效

---

## 10. 验收标准（详见 checklist.md）

- 库能解析真实 GGUF 文件（如 Llama/Qwen 的小量化版）的完整元数据
- CLI 默认文本输出与 `--json` 输出均正确
- 所有 13 种 KV 类型 + 数组正确解析
- 张量描述符（名称/形状/类型/偏移/元素数）正确
- 损坏/截断文件不 panic，返回合适错误
- `cargo build` / `cargo test` / `cargo clippy` 通过，无 warning
