# GGUF 推理 GUI 设计规范

## 1. 概述

为 `gguf` crate 新增一个 **GUI 可执行文件**（`gguf-gui`），覆盖现有全部命令行功能：

- **推理**（对应 `gguf-infer`）：prompt 补全、多轮对话、流式输出、采样参数调节
- **模型查看**（对应 `gguf-dump`）：文件摘要、KV 元数据、张量列表

GUI 基于 **egui (eframe)** 实现，纯 Rust 无 Web 依赖，编译为单个 exe，跨平台（Windows / Linux / macOS）。

### 设计原则

1. **功能等价**：所有命令行参数在 GUI 中都有对应控件
2. **线程安全**：推理在后台线程执行，UI 不卡顿
3. **增量集成**：复用现有 `gguf` 库 API（`Engine`、`GgufFile`、`Tokenizer`），不修改核心逻辑
4. **流式输出**：token 生成实时追加到界面

---

## 2. 依赖与构建

### 2.1 新增依赖（Cargo.toml）

```toml
# GUI（egui / eframe）
eframe = { version = "0.29", optional = true }
egui = { version = "0.29", optional = true }

# 后台推理线程
# 使用 std::thread（无需额外依赖）
```

### 2.2 Feature 配置

```toml
[features]
gui = ["dep:eframe", "dep:egui"]

# default 不含 gui，避免库用户强制编译 GUI 依赖
default = ["mmap", "json", "parallel"]
```

### 2.3 新增二进制目标

```toml
[[bin]]
name = "gguf-gui"
path = "src/bin/gguf_gui.rs"
required-features = ["gui"]
```

构建命令：
```bash
cargo build --release --features gui
```

---

## 3. 应用架构

### 3.1 模块结构

```
src/bin/gguf_gui.rs          # 入口：eframe::run_native
src/gui/                     # GUI 模块（仅 gui feature 编译）
├── mod.rs                   # 模块声明 + 共享类型
├── app.rs                   # 主 App 结构体 + 标签页管理
├── chat_view.rs             # 对话视图
├── prompt_view.rs           # Prompt 补全视图
├── model_view.rs            # 模型查看视图
├── settings.rs              # 采样参数设置面板
├── inference.rs             # 后台推理线程 + 消息通道
└── state.rs                 # 应用全局状态
```

> 注意：`src/gui/` 作为 bin 的内部模块组织，不加入 lib。
> 若需被其他 bin 复用，后续可迁移到 `src/gui/` 并由 lib 导出（当前阶段不需要）。

### 3.2 线程模型

```
┌─────────────────────────────────────────────────────┐
│                    UI Thread (main)                  │
│  eframe run → App::update → 渲染 → 处理消息         │
│                                                     │
│  ┌─────────────┐    mpsc::Receiver     ┌─────────┐ │
│  │  State      │◄─────────────────────│  消息    │ │
│  │  (数据)      │                      │  接收    │ │
│  └─────────────┘                      └─────────┘ │
└─────────────────────────────────────────────────────┘
                     ▲
                     │ mpsc::Channel
                     │ (InferMsg)
                     ▼
┌─────────────────────────────────────────────────────┐
│               Inference Thread (worker)             │
│  Engine 持有 → generate/chat 循环 → 流式发送 token  │
│                                                     │
│  ┌─────────────┐    mpsc::Sender       ┌─────────┐ │
│  │  Engine     │──────────────────────►│ InferMsg│ │
│  │  (模型)      │                       │  发送    │ │
│  └─────────────┘                       └─────────┘ │
└─────────────────────────────────────────────────────┘
```

**关键约束**：
- `Engine` 仅在推理线程中使用（`&mut self` 方法）
- UI 线程通过 `mpsc::Sender<UiCommand>` 发送命令
- 推理线程通过 `mpsc::Sender<InferMsg>` 发送结果
- 消息均为可 Send 类型（`String`、`u32`、`f32` 等）

### 3.3 消息类型

```rust
/// UI → 推理线程
enum UiCommand {
    /// 加载模型
    LoadModel { path: String, sampler: SamplerConfig },
    /// 发送 prompt（单轮补全）
    Prompt { text: String, max_tokens: usize },
    /// 发送对话消息（多轮）
    Chat { text: String, max_tokens: usize },
    /// 重置对话上下文
    Reset,
    /// 停止当前生成
    Stop,
    /// 退出
    Quit,
}

/// 推理线程 → UI
enum InferMsg {
    /// 模型加载完成
    ModelLoaded { name: String, arch: String, layers: u32, embed_dim: u32,
                  vocab_size: u32, size_mb: f64, load_ms: u128 },
    /// 模型加载失败
    LoadError { message: String },
    /// 流式 token
    Token { id: u32, text: String },
    /// 一轮生成完成
    Done { full_text: String, elapsed_ms: u128, ctx_len: usize, ctx_limit: usize },
    /// 生成出错
    Error { message: String },
    /// 已停止
    Stopped,
    /// 已重置
    ResetDone,
}
```

---

## 4. UI 布局

### 4.1 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│  [工具栏]                                                    │
│  📂 选择模型  [模型名称: xxx.gguf ▼]  [加载]  [状态: 就绪/生成中] │
├─────────────────────────────────────────────────────────────┤
│  [标签页]                                                    │
│  ┌──────────┬──────────────┬──────────────┐                  │
│  │ 对话      │ Prompt 补全   │ 模型信息      │                  │
│  └──────────┴──────────────┴──────────────┘                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │              [当前标签页内容区]                        │    │
│  │                                                     │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  [设置面板] (可折叠)                                   │    │
│  │  温度 [0.8]  Top-K [40]  Top-P [0.95]  Min-P [0.0]  │    │
│  │  重复惩罚 [1.1]  种子 [0]  最大Token [512]  ☐ 贪心    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  [状态栏]                                                    │
│  ctx: 128/32768  |  生成: 45 tok/s  |  耗时: 2.8s  |  ✅ 就绪 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 标签页 1：对话（Chat）

对应 `gguf-infer --chat`。

```
┌─────────────────────────────────────────────────────────┐
│  对话区 (ScrollArea, 自动滚动到底部)                       │
│                                                         │
│  ┌─ 用户 ──────────────────────────────────────────┐   │
│  │ 你好，请介绍一下你自己                              │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─ 助手 ──────────────────────────────────────────┐   │
│  │ 你好！我是Qwen，由阿里云创建的大语言模型。          │   │
│  │ 我可以回答各类问题、协助写作、编程等。              │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─ 用户 ──────────────────────────────────────────┐   │
│  │ 1+1等于几？                                       │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─ 助手 ──────────────────────────────────────────┐   │
│  │ 1+1等于2。                                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [生成中... ●]  ← 流式 token 实时追加                    │
├─────────────────────────────────────────────────────────┤
│  输入区                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ [多行文本框，Enter 发送，Shift+Enter 换行]         │   │
│  └─────────────────────────────────────────────────┘   │
│  [发送]  [重置上下文]  [停止生成]                         │
└─────────────────────────────────────────────────────────┘
```

**行为**：
- 模型加载后可发送消息
- 发送后调用 `Engine::chat()`，流式 token 实时追加到助手气泡
- 上下文通过 KV cache 自动累积（与命令行 `--chat` 一致）
- `重置上下文` 按钮调用 `Engine::reset()`
- `停止生成` 按钮发送 `Stop` 命令，中断当前生成
- 生成中输入框禁用，停止后可重新输入
- 对话历史在 UI 中保留（切换标签页不丢失）

### 4.3 标签页 2：Prompt 补全

对应 `gguf-infer --prompt`。

```
┌─────────────────────────────────────────────────────────┐
│  Prompt 输入区                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ [多行文本框]                                      │   │
│  │ 请解释什么是量子计算...                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  选项: ☐ 流式输出  ☐ 贪心解码                              │
├─────────────────────────────────────────────────────────┤
│  输出区 (ScrollArea)                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 量子计算是利用量子力学原理...                       │   │
│  │ [流式追加中...]                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  统计: 耗时 2.8s | 45 tok/s | ctx 128/32768             │
├─────────────────────────────────────────────────────────┤
│  [运行]  [清除输出]                                       │
└─────────────────────────────────────────────────────────┘
```

**行为**：
- 每次运行调用 `Engine::generate()`（无状态，position 0 全量 prefill）
- 流式/非流式切换（对应 `--no-stream`）
- 贪心解码勾选后覆盖温度/top-k/top-p（对应 `--greedy`）
- 显示耗时、token 速率、ctx 用量（对应 `--verbose`）

### 4.4 标签页 3：模型信息

对应 `gguf-dump`。

```
┌─────────────────────────────────────────────────────────┐
│  文件摘要                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 文件名: qwen2.5-0.5b.gguf                        │   │
│  │ 架构: qwen2                                     │   │
│  │ 模型名: Qwen2.5-0.5B-Instruct                     │   │
│  │ GGUF 版本: 3                                     │   │
│  │ 对齐: 32                                         │   │
│  │ 数据偏移: 1048576                                │   │
│  │ 文件大小: 468.6 MB                               │   │
│  │ 张量数: 278                                      │   │
│  │ KV 数: 42                                        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  KV 元数据 (表格, 可搜索/过滤)                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 键                          │ 类型   │ 值         │   │
│  │ general.architecture        │ str    │ qwen2     │   │
│  │ general.name                │ str    │ Qwen2.5.. │   │
│  │ qwen2.block_count           │ u32    │ 24        │   │
│  │ qwen2.context_length        │ u32    │ 32768     │   │
│  │ ...                         │        │           │   │
│  └─────────────────────────────────────────────────┘   │
│  [搜索框: ________]  ☐ 显示全部                          │
│                                                         │
│  张量列表 (表格)                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 名称                          │ 形状        │ 类型   │   │
│  │ token_embd.weight             │ [896,151936]│ F16   │   │
│  │ blk.0.attn_q.weight           │ [896,896]   │ Q4_K  │   │
│  │ blk.0.attn_k.weight           │ [896,128]   │ Q4_K  │   │
│  │ ...                           │             │       │   │
│  └─────────────────────────────────────────────────┘   │
│  ☐ 显示全部 (默认前 50)                                   │
│                                                         │
│  词表查询                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │ [Token ID: ____] [Token 字符串: ____] [查询]      │   │
│  │ 结果: ID=151645  字符串="im_end"                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  导出: [JSON] [复制摘要]                                   │
└─────────────────────────────────────────────────────────┘
```

**行为**：
- 模型加载后自动填充摘要
- KV 表格支持搜索过滤（对应 `--key`）
- 张量列表默认前 50，可展开全部（对应 `--tensors-all`）
- 词表查询支持按 ID 和按字符串（对应 `--token-id` / `--token-str`）
- JSON 导出（对应 `--json --pretty`）
- 复制摘要到剪贴板

### 4.5 设置面板（底部可折叠）

对应 `gguf-infer` 所有采样参数：

| 控件 | 类型 | 参数 | 默认值 | 范围 |
|------|------|------|--------|------|
| 温度 | Slider + TextEdit | `--temperature` | 0.8 | 0.0–2.0 |
| Top-K | DragValue | `--top-k` | 40 | 0–1000 |
| Top-P | Slider | `--top-p` | 0.95 | 0.0–1.0 |
| Min-P | Slider | `--min-p` | 0.0 | 0.0–1.0 |
| 重复惩罚 | Slider | `--repeat-penalty` | 1.1 | 1.0–2.0 |
| 种子 | TextEdit | `--seed` | 0 | 0–u64 |
| 最大 Token | DragValue | `--max-tokens` | 512 | 1–32768 |
| 贪心 | Checkbox | `--greedy` | false | — |

**贪心勾选时**：温度/top-k/top-p/min-p 控件禁用，采样器配置强制为 `temperature=0`。

**参数修改时机**：
- 修改后**立即生效**于后续生成（无需重载模型）
- 推理线程在每次 `generate`/`chat` 前读取最新 `SamplerConfig`
- 若 `seed` 变更，推理线程重置 RNG

### 4.6 工具栏

| 控件 | 功能 |
|------|------|
| 📂 选择模型 | 打开文件对话框，过滤 `*.gguf` |
| 模型名称下拉 | 显示当前加载的模型（可切换最近使用的） |
| [加载] | 加载选中模型到推理线程 |
| [卸载] | 释放模型内存 |
| 状态指示 | 🟢 就绪 / 🟡 生成中 / 🔴 错误 |

**文件对话框**：egui 无内置文件对话框，使用 `rfd` crate（轻量跨平台）。

```toml
rfd = { version = "0.15", optional = true }  # gui feature
```

### 4.7 状态栏

```
ctx: 128/32768  |  45 tok/s  |  2.8s  |  模型: qwen2.5-0.5b  |  ✅ 就绪
```

实时更新：
- `ctx`：当前 KV cache 长度 / 上下文上限
- `tok/s`：最近一轮生成的 token 速率
- 耗时：最近一轮生成耗时
- 模型名：当前加载的模型

---

## 5. 推理线程设计

### 5.1 线程生命周期

```
UI 启动
  │
  ├── 创建推理线程 (idle 状态，等待命令)
  │
  ├── 用户点击"加载"
  │     │
  │     ├── UI 发送 LoadModel { path, sampler }
  │     │
  │     ├── 推理线程: GgufFile::from_path(path)
  │     ├── 推理线程: Engine::new(file, sampler_config)
  │     ├── 推理线程: 发送 ModelLoaded { ... }
  │     │
  │     └── UI: 状态→就绪，填充模型信息
  │
  ├── 用户发送消息
  │     │
  │     ├── UI 发送 Chat { text, max_tokens }
  │     │
  │     ├── 推理线程: engine.chat(text, max_tokens, |id, txt| {
  │     │     sender.send(InferMsg::Token { id, text: txt.to_string() })
  │     │   })
  │     ├── 推理线程: 发送 Done { full_text, elapsed, ctx }
  │     │
  │     └── UI: 流式追加 token，完成后显示统计
  │
  └── 用户点击"停止"
        │
        ├── UI 发送 Stop
        │
        ├── 推理线程: 标记停止（通过 channel 通知）
        │
        └── 推理线程: 发送 Stopped
```

### 5.2 停止机制

`Engine::chat` / `Engine::generate` 的 `on_token` 闭包中检查停止标志：

```rust
// 推理线程内
let stop_flag = Arc::new(AtomicBool::new(false));
let stop_clone = stop_flag.clone();

let _ = engine.chat(text, max_tokens, |id, txt| {
    if stop_clone.load(Ordering::Relaxed) {
        // 返回 false 表示中止（需 Engine 支持）
        return false;  // 注意：当前 on_token 返回 ()，需改造
    }
    sender.send(InferMsg::Token { id, text: txt.to_string() }).ok();
    true
});
```

**重要**：当前 `Engine::generate` 和 `Engine::chat` 的 `on_token` 回调签名为 `FnMut(u32, &str)`（返回 `()`），**不支持提前终止**。

**解决方案**（不修改 Engine API）：
- 在推理线程中用 `mpsc::channel` 作为停止信号
- `on_token` 闭包中 `try_recv` 检查停止消息
- 若收到停止消息，设置标志位，后续闭包调用直接 `return`（但仍会生成到 max_tokens）
- **更好的方案**：在 Engine 中新增 `generate_with_cancel` 方法，`on_token` 返回 `bool`（false = 中止）

> 决策：新增 `Engine::generate_cancellable` 和 `Engine::chat_cancellable` 方法，
> `on_token: F: FnMut(u32, &str) -> bool`，返回 `false` 时中止生成。
> 这是最小化改动，不影响现有 API。

### 5.3 SamplerConfig 动态更新

推理线程持有 `Sampler`，但 `SamplerConfig` 在 UI 中可实时修改。

**方案**：
- 推理线程每次 `generate`/`chat` 前从 `UiCommand` 中携带的最新 `SamplerConfig` 更新
- `Sampler` 新增 `set_config(&mut self, config: SamplerConfig)` 方法
- 若 `seed` 变更，重置 RNG

---

## 6. 与命令行的功能映射

| 命令行参数/功能 | GUI 对应 |
|----------------|---------|
| `gguf-infer model.gguf` | 工具栏 📂 选择模型 + [加载] |
| `--prompt "text"` / `-p` | Prompt 补全标签页 输入框 |
| `--max-tokens` / `-n` | 设置面板 最大Token |
| `--temperature` / `-t` | 设置面板 温度 |
| `--top-k` | 设置面板 Top-K |
| `--top-p` | 设置面板 Top-P |
| `--min-p` | 设置面板 Min-P |
| `--repeat-penalty` | 设置面板 重复惩罚 |
| `--seed` / `-s` | 设置面板 种子 |
| `--greedy` | 设置面板 贪心 Checkbox |
| `--no-stream` | Prompt 补全标签页 流式 Checkbox（取消勾选=非流式） |
| `--verbose` / `-v` | 状态栏始终显示统计 |
| `--chat` | 对话标签页 |
| `:reset` | 对话标签页 [重置上下文] 按钮 |
| `:quit` / `:q` | 关闭窗口 / [退出] |
| 流式输出 | 对话/Prompt 标签页 token 实时追加 |
| `gguf-dump model.gguf` | 模型信息标签页 |
| `gguf-dump -j` | 模型信息 [JSON 导出] |
| `gguf-dump -k key` | 模型信息 KV 搜索框 |
| `gguf-dump -t` | 模型信息 张量列表 ☐ 显示全部 |
| `gguf-dump -T id` | 模型信息 词表查询 Token ID |
| `gguf-dump -S str` | 模型信息 词表查询 Token 字符串 |

---

## 7. 需要修改的库 API

### 7.1 `Engine` 新增可取消生成方法

```rust
impl Engine {
    /// 可取消的 prompt 补全。
    /// `on_token` 返回 `false` 时中止生成，返回已生成的完整文本。
    pub fn generate_cancellable<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> GgufResult<String>
    where
        F: FnMut(u32, &str) -> bool,
    { /* 同 generate，但 on_token 返回 false 时 break */ }

    /// 可取消的多轮对话。
    pub fn chat_cancellable<F>(
        &mut self,
        text: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> GgufResult<String>
    where
        F: FnMut(u32, &str) -> bool,
    { /* 同 chat，但 on_token 返回 false 时 break */ }
}
```

### 7.2 `Sampler` 新增配置更新方法

```rust
impl Sampler {
    /// 更新采样配置（seed 变更时重置 RNG）。
    pub fn set_config(&mut self, config: SamplerConfig) {
        if config.seed != self.config.seed && config.seed != 0 {
            self.rng = StdRng::seed_from_u64(config.seed);
        }
        self.config = config;
    }
}
```

### 7.3 现有 `generate` / `chat` 保持不变

`generate_cancellable` 内部可复用 `generate` 的逻辑（仅回调签名不同），
或直接复制一份修改。为最小化改动，建议**复制并修改**。

---

## 8. 错误处理

| 错误场景 | UI 表现 |
|---------|---------|
| 模型文件不存在 | 工具栏状态 🔴 + 弹窗提示 |
| GGUF 格式无效 | 状态 🔴 + 错误信息（InvalidMagic 等） |
| 架构不支持 | 状态 🔴 + "不支持的架构: xxx" |
| 张量缺失 | 状态 🔴 + "缺失张量: xxx" |
| Tokenizer 错误 | 状态 🔴 + 错误详情 |
| 上下文超出 | 对话区显示 ⚠️ "上下文超出，请重置" + [重置] 按钮 |
| 生成中断（用户停止） | 状态栏 "已停止" |
| 推理线程 panic | 捕获 panic，状态 🔴 + "推理线程异常" |

### 推理线程 panic 保护

```rust
let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    // 推理循环
}));
if result.is_err() {
    sender.send(InferMsg::Error { message: "推理线程 panic".into() }).ok();
}
```

---

## 9. 性能考虑

| 方面 | 策略 |
|------|------|
| 模型加载 | 后台线程，UI 显示进度（"加载中..."） |
| 推理 | 后台线程，`rayon` 并行（已有 `parallel` feature） |
| 流式输出 | mpsc channel，UI 每帧 drain 消息批量更新 |
| 大文件 mmap | `GgufFile::from_path` 已优先 mmap |
| 张量物化 | `LlamaModel` 内部 `weight_cache`（已有 Arc 缓存） |
| UI 重绘 | egui 仅在有消息或用户操作时重绘（`ctx.request_repaint`） |

### 消息批量处理

UI 线程每帧 `update()` 中：
```rust
// 批量 drain 消息，减少重绘次数
while let Ok(msg) = rx.try_recv() {
    self.handle_infer_msg(msg);
}
```

---

## 10. 跨平台兼容

| 平台 | 支持 | 说明 |
|------|------|------|
| Windows | ✅ | 主平台，`rfd` 用 COM 文件对话框 |
| Linux | ✅ | `rfd` 用 GTK/zenity |
| macOS | ✅ | `rfd` 用 NSOpenPanel |

egui 0.29 + eframe 0.29 在三平台均可编译。`windows-sys` 依赖已通过 `[target.'cfg(windows)'.dependencies]` 隔离。

---

## 11. 验收标准概要

详见 `checklist.md`。核心验收点：

1. `cargo build --release --features gui` 编译成功，生成 `gguf-gui.exe`
2. 启动 GUI 后选择 `.gguf` 文件可成功加载模型
3. 对话标签页可多轮交互，流式输出，上下文累积
4. Prompt 标签页可单轮补全，流式/非流式可切换
5. 模型信息标签页显示摘要、KV、张量、词表查询
6. 设置面板所有参数可调节且即时生效
7. 停止生成按钮可中断当前推理
8. 重置上下文按钮可清空对话
9. 模型信息可导出 JSON
10. 所有现有 192 个测试仍通过（库 API 兼容性）
