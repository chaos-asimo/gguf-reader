# GGUF 元数据读取程序 — 任务拆解 (Tasks)

> 分 6 个阶段，自底向上。每阶段可独立编译/测试。勾选 [x] 表示完成。

## 阶段 1：项目脚手架

- [x] T1.1 初始化 Cargo 项目（`cargo init --name gguf`），配置 workspace
- [x] T1.2 创建 `src/lib.rs` 并 re-export 公共类型
- [x] T1.3 配置 `Cargo.toml` 依赖：`serde`(derive)、`serde_json`、`clap`(derive)、`anyhow`、`memmap2`(feature `mmap`)
- [x] T1.4 建立目录结构：`src/{types,tensor,header,file,cursor,error}.rs`、`src/bin/gguf_dump.rs`、`tests/`
- [x] T1.5 创建 feature 标志：`default = ["mmap"]`，`mmap = ["dep:memmap2"]`
- [x] 验证：`cargo build` 通过（空库）

## 阶段 2：错误类型与字节读取器

- [x] T2.1 实现 `error.rs`：`GgufError` 枚举（全部变体）+ `Display` + `std::error::Error` + `From<io::Error>`
- [x] T2.2 实现 `cursor.rs`：`Cursor<'a>` 带边界保护
  - [x] 读取 u8/i8/u16/i16/u32/i32/u64/i64/f32/f64（小端）
  - [x] 读取 bool（i8）
  - [x] 读取 string（uint64 长度 + UTF-8 校验）
  - [x] `remaining()` / `pos()`
  - [x] 每次读取前边界校验，越界返回 `OutOfBounds{offset, required, file_size}`
- [x] T2.3 单元测试：Cursor 各类型正确读取；越界报错；短缓冲报错
- [x] 验证：`cargo test` 通过

## 阶段 3：类型定义与值模型

- [x] T3.1 实现 `types.rs`：
  - [x] `GgufType` 枚举（13 值）+ `from_i32` + `Display`（类型名）
  - [x] `GgmlType` 枚举（常见量化值 + `Unknown(i32)`) + `from_i32` + `Display`
  - [x] `GgufArray` 结构（elem_type + data）
  - [x] `GgufValue` 枚举（全部变体）+ `value_type()` + `as_*` 便捷方法 + `display()`
- [x] T3.2 实现 `header.rs`：`GgufHeader` 结构（magic/version/n_tensors/n_kv）
- [x] T3.3 实现 `tensor.rs`：`TensorInfo` 结构 + `num_elements` + `est_element_size` + `est_data_size`
- [x] T3.4 单元测试：
  - [x] `GgufType::from_i32` 全部映射
  - [x] `GgmlType::from_i32` 常见值 + 未知值降级
  - [x] `GgufValue::display()` 各类型（含数组）
  - [x] `TensorInfo::num_elements` / `est_data_size`
- [x] 验证：`cargo test` 通过

## 阶段 4：核心解析（GgufFile）

- [x] T4.1 实现 `file.rs`：
  - [x] `GgufFile` 结构（header/kv/tensors/alignment/data_offset/file_size）
  - [x] `from_buffer(&[u8])`：
    1. 解析 header（magic/version/n_tensors/n_kv）
    2. 循环 n_kv 解析 KV（含数组递归解析，防嵌套）
    3. 循环 n_tensors 解析张量描述符
    4. 读取 alignment（general.alignment 缺省 32）
    5. 计算 data_offset（对齐填充）
  - [x] `from_reader<R: Read+Seek>`：整体读入 Vec<u8> 后调 from_buffer
  - [x] `from_path`：mmap（feature）→ 读元数据区克隆 → from_buffer；mmap 失败回退 from_reader
  - [x] 便捷方法：`get`/`find_tensor`/`architecture`/`model_name`/`kv_map`
- [x] T4.2 数组解析预检：`count * elem_min_size ≤ remaining`，否则 `OutOfBounds`
- [x] T4.3 集成测试 `tests/parse_buffer.rs`：
  - [x] 构造内存 GGUF 缓冲工具函数（header+kv+tensors+假数据体）
  - [x] 覆盖全部 13 种 KV 类型
  - [x] 覆盖字符串、负数 int、f32/f64、bool
  - [x] 覆盖数组（含大 count 截断预检）
  - [x] 覆盖多张量、不同 shape、未知 dtype
  - [x] 断言 header/kv/tensors/alignment/data_offset 全部正确
- [x] T4.4 健壮性测试：
  - [x] 错误 magic/version
  - [x] 任意截断
  - [x] 损坏数组 count
  - [x] 嵌套数组
  - [x] 空/短文件
- [x] 验证：`cargo test` 全部通过

## 阶段 5：CLI 工具

- [x] T5.1 实现 `src/bin/gguf_dump.rs`：
  - [x] clap 参数（PATH 位置参数 + 全部选项）
  - [x] 读取文件 → `GgufFile::from_path`
  - [x] 文本输出渲染（summary / kv / tensors 表格）
  - [x] JSON 输出（serde，大数组截断）
  - [x] 退出码映射（0/1/2/3/4）
  - [x] 人类可读大小格式化（B/KB/MB/GB）
  - [x] 张量表格列对齐
- [x] T5.2 实现 `--summary-only` / `-k <KEY>` / `--quiet` 过滤
- [x] T5.3 实现 JSON 大数组截断逻辑（阈值 1000，truncated + total）
- [x] T5.4 CLI 集成测试 `tests/cli.rs`：
  - [x] 文本输出含关键键
  - [x] `--json` 可反序列化且字段正确
  - [x] 退出码：不存在=1、魔数错=2、损坏=4
  - [x] 大数组截断生效
- [x] 验证：`cargo test` + 手动运行 `cargo run -- <真实.gguf>` 通过

## 阶段 6：质量收尾

- [x] T6.1 `cargo clippy --all-targets` 清零 warning
- [x] T6.2 `cargo fmt` 格式化
- [x] T6.3 移除生产代码中所有 `unwrap`/`expect`/`panic!`（测试除外）
- [x] T6.4 补充 `Cargo.toml` 描述、license、version、repository 元信息
- [x] T6.5 用一个真实小型 GGUF 文件端到端验证（解析 + 文本 + JSON）
  （gist-embedding-v0.Q2_K.gguf，54MB，architecture=bert/12 块/30522 词表，197 张量，data_offset=760,992）
- [x] T6.6 核对 checklist.md 全部项完成（A-I 组 60 项全部 [x]）
- [x] 验证：`cargo build --release` + `cargo test` + `cargo clippy` 全绿

## 阶段依赖关系

```
T1 → T2 → T3 → T4 → T5 → T6
```

- T2/T3 可在 T1 后部分并行（error 与 types 无相互依赖，cursor 依赖 error）
- T4 依赖 T2（cursor）+ T3（types）
- T5 依赖 T4
- T6 收尾

## 关键技术风险与对策

| 风险 | 对策 |
|------|------|
| 数组 count 损坏导致 OOM | 解析前 `count * elem_min_size ≤ remaining` 预检（T4.2）|
| mmap 跨平台差异 | memmap2 封装；feature 可选；失败回退普通读（T4.1）|
| 大模型文件 I/O 慢 | 只读元数据区（文件前部），不加载数据体（T4.1/T5）|
| 未知 dtype/枚举崩溃 | GgmlType::Unknown 降级；GgufType 未知报错（T3.1）|
| UTF-8 非法字节 panic | string 读取用 `from_utf8` 校验，失败返回 Err（T2.2）|
