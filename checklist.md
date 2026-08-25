# GGUF 元数据读取程序 — 验收清单 (Checklist)

> 实现完成后逐项核对。每项含 [ ] 待办 / [x] 完成标记。

## A. 构建与质量

- [x] A1. `cargo build` 成功，无 error
- [x] A2. `cargo build --release` 成功
- [x] A3. `cargo test` 全部通过（54 个：28 lib + 10 parse_buffer + 16 cli + 1 其他）
- [x] A4. `cargo clippy --all-targets` 无 warning
- [x] A5. `cargo fmt --check` 格式合规
- [x] A6. 库 crate 核心解析路径零第三方依赖（memmap2 为可选 feature）

## B. 文件头解析

- [x] B1. 正确读取 magic（4 字节）并校验等于 `0x46554747`，不符返回 `InvalidMagic`
- [x] B2. 正确读取 version（uint32），非 3 返回 `UnsupportedVersion`
- [x] B3. 正确读取 n_tensors（int64 → u64），负值/超大值报错
- [x] B4. 正确读取 n_kv（int64 → u64），负值/超大值报错

## C. KV 元数据解析（全部 13 种类型）

- [x] C1. UINT8 解析正确
- [x] C2. INT8 解析正确
- [x] C3. UINT16 解析正确
- [x] C4. INT16 解析正确
- [x] C5. UINT32 解析正确
- [x] C6. INT32 解析正确（含负值）
- [x] C7. FLOAT32 解析正确
- [x] C8. BOOL 解析正确（i8 → bool）
- [x] C9. STRING 解析正确（长度前缀 + UTF-8，无 null 终止）
- [x] C10. ARRAY 解析正确（元素类型 + count + 元素序列）
- [x] C11. UINT64 解析正确
- [x] C12. INT64 解析正确（含负值）
- [x] C13. FLOAT64 解析正确
- [x] C14. 数组元素类型若为 ARRAY（嵌套）返回 `InvalidArrayElemType`
- [x] C15. 未知 GGUF_TYPE 值返回 `InvalidGgufType`
- [x] C16. 字符串长度超出剩余字节返回 `OutOfBounds`
- [x] C17. KV 保持文件内出现顺序
- [x] C18. 大数组（count 巨大）在解析前预检字节数，防 OOM

## D. 张量描述符解析

- [x] D1. 张量名（string）解析正确
- [x] D2. n_dims（uint32）解析正确
- [x] D3. 各维度 ne[]（int64）按 `ne[n_dims-1-i]` 还原为逻辑顺序
- [x] D4. dtype（ggml_type）解析正确，常见值具名
- [x] D5. 未知 dtype 值降级为 `GgmlType::Unknown(n)` 不崩溃
- [x] D6. offset（uint64）解析正确
- [x] D7. `TensorInfo::num_elements` 计算正确（shape 各维之积）
- [x] D8. `TensorInfo::est_data_size` 对 F32/F16/BF16 等精确类型估算正确
- [x] D9. 维度为负返回 `InvalidTensorDim`
- [x] D10. 张量保持文件内出现顺序

## E. 顶层结构（GgufFile）

- [x] E1. `from_buffer` 解析成功且字段完整
- [x] E2. `from_path` 优先 mmap，失败回退普通读取
- [x] E3. `from_reader` 从 Read+Seek 解析成功
- [x] E4. `alignment` 正确读取（general.alignment 缺省 32）
- [x] E5. `data_offset` 正确计算（含对齐填充到 alignment 边界）
- [x] E6. `file_size` 记录正确
- [x] E7. `get(key)` 按键查找正确，未命中返回 None
- [x] E8. `find_tensor(name)` 按名查找正确
- [x] E9. `architecture()` / `model_name()` 便捷访问正确
- [x] E10. `kv_map()` 返回正确 HashMap

## F. CLI 工具（gguf-dump）

- [x] F1. `gguf-dump <PATH>` 默认文本输出成功
- [x] F2. 文本输出包含：文件路径、大小、版本、张量数、KV 数、对齐、数据偏移、架构、模型名
- [x] F3. 文本输出 KV 列表带类型标签与值
- [x] F4. 文本输出张量列表（名称/形状/类型/偏移/大小），默认截断前 50
- [x] F5. `--json` 输出合法 JSON（可被 serde_json 反序列化）
- [x] F6. `--pretty` 美化 JSON
- [x] F7. `-k <KEY>` 仅显示指定键
- [x] F8. `--summary-only` 仅摘要
- [x] F9. 退出码：成功=0、文件不存在=1、魔数错=2、版本错=3、损坏=4
- [x] F10. 大数组 JSON 输出截断生效（>1000 元素时 truncated=true + total）
- [x] F11. 人类可读大小格式（KB/MB/GB）正确

## G. 健壮性（损坏/异常文件）

- [x] G1. 空文件 → 合适错误（OutOfBounds），不 panic
- [x] G2. 短文件（仅 magic）→ 错误，不 panic
- [x] G3. 错误 magic → `InvalidMagic`，CLI 退出码 2
- [x] G4. 错误 version → `UnsupportedVersion`，CLI 退出码 3
- [x] G5. 任意位置截断 → `OutOfBounds`，CLI 退出码 4
- [x] G6. 损坏数组 count → 报错不 OOM
- [x] G7. 非法 UTF-8 字符串 → 报错不 panic
- [x] G8. 所有错误路径返回 `Err`，无 `unwrap`/`expect`/`panic!` 在生产代码

## H. 真实文件验证

- [x] H1. 对一个真实小型 GGUF 文件（如 ≤1GB 量化模型）成功解析
  （gist-embedding-v0.Q2_K.gguf，54,236,512 字节，下载自 HuggingFace）
- [x] H2. 解析出的架构、块数、词表大小等与已知值一致
  （architecture=bert、block_count=12、context_length=512、embedding_length=768、
  feed_forward_length=3072、head_count=12、tokens=30522，均与模型已知规格一致）
- [x] H3. 张量总数、首张量形状与已知一致（197 个张量；首张量 token_embd_norm.bias [768]）
- [x] H4. data_offset 不超过 file_size（760,992 < 54,236,512）

## I. 文档与规范一致性

- [x] I1. 代码结构与 spec.md 第 6 节项目结构一致
- [x] I2. 公共 API 与 spec.md 第 4 节一致
- [x] I3. 错误类型与 spec.md 第 4.2 节一致
- [x] I4. CLI 参数与 spec.md 第 5.1 节一致
- [x] I5. `Cargo.toml` 描述/版本/feature 配置完整
