# GGUF 推理 GUI 任务分解

## 任务总览

| 阶段 | 任务数 | 说明 |
|------|--------|------|
| Phase 1: 基础架构 | 3 | 依赖配置、库 API 扩展、模块骨架 |
| Phase 2: 推理线程 | 2 | 后台线程、消息通道 |
| Phase 3: UI 框架 | 2 | 主 App、标签页管理 |
| Phase 4: 功能视图 | 4 | 对话、Prompt、模型信息、设置 |
| Phase 5: 集成测试 | 1 | 端到端验证 |

**总计：12 个任务**

---

## Phase 1: 基础架构

### T1: 依赖配置与 feature 扩展
- **文件**: `Cargo.toml`
- **改动**:
  - 添加 `eframe = { version = "0.29", optional = true }`
  - 添加 `egui = { version = "0.29", optional = true }`
  - 添加 `rfd = { version = "0.15", optional = true }`
  - 添加 `feature: gui = ["dep:eframe", "dep:egui", "dep:rfd"]`
  - 添加 `[[bin]] name = "gguf-gui" path = "src/bin/gguf_gui.rs" required-features = ["gui"]`
- **验证**: `cargo check --features gui` 通过；`cargo check`（无 gui）仍通过

### T2: 库 API 扩展 — Engine 可取消方法
- **文件**: `src/infer/engine.rs`
- **改动**:
  - 新增 `generate_cancellable<F: FnMut(u32, &str) -> bool>` 方法
  - 新增 `chat_cancellable<F: FnMut(u32, &str) -> bool>` 方法
  - `on_token` 返回 `false` 时 break 采样循环
  - 返回已生成的完整文本
- **验证**: 单元测试（on_token 返回 false 时提前终止）；现有测试仍通过

### T3: 库 API 扩展 — Sampler 配置更新
- **文件**: `src/infer/sampler.rs`
- **改动**:
  - 新增 `set_config(&mut self, config: SamplerConfig)` 方法
  - seed 变更时重置 RNG
- **验证**: 单元测试；现有测试仍通过

---

## Phase 2: 推理线程

### T4: 后台推理线程与消息通道
- **文件**: `src/bin/gguf_gui.rs`（新建）、`src/gui/inference.rs`（新建）
- **改动**:
  - 定义 `UiCommand` 枚举（LoadModel/Prompt/Chat/Reset/Stop/Quit）
  - 定义 `InferMsg` 枚举（ModelLoaded/LoadError/Token/Done/Error/Stopped/ResetDone）
  - 创建推理线程，持有 `GgufFile` + `Engine`
  - 线程循环：`recv` UiCommand → 执行 → `send` InferMsg
  - 流式 token 通过 `InferMsg::Token` 逐条发送
  - `catch_unwind` 保护 panic
  - `SamplerConfig` 动态更新（每次 generate/chat 前 set_config）
- **验证**: 线程可正常加载模型、生成 token、停止、重置

### T5: 模型信息数据结构
- **文件**: `src/gui/state.rs`（新建）
- **改动**:
  - `ModelInfo` 结构：name、arch、model_name、gguf_version、alignment、data_offset、file_size、tensor_count、kv_count
  - `ChatMessage` 结构：role（user/assistant）、content
  - `AppState` 枚举：Idle/Loading/Ready/Generating/Error
  - `GenStats` 结构：elapsed_ms、tokens、tok_per_s、ctx_len、ctx_limit
- **验证**: 编译通过

---

## Phase 3: UI 框架

### T6: 主 App 结构与标签页管理
- **文件**: `src/gui/app.rs`（新建）、`src/gui/mod.rs`（新建）
- **改动**:
  - `GgufApp` 实现 `eframe::App` trait
  - `new()` 初始化：创建推理线程、建立 mpsc 通道
  - `update()` 主循环：drain 消息 → 更新状态 → 渲染
  - 工具栏渲染（📂 选择模型、加载、卸载、状态指示）
  - 标签页（对话 / Prompt 补全 / 模型信息）
  - 状态栏渲染（ctx、tok/s、耗时、模型名）
  - 文件对话框（`rfd::FileDialog`）
- **验证**: GUI 可启动，标签页可切换，工具栏正常

### T7: egui 入口
- **文件**: `src/bin/gguf_gui.rs`
- **改动**:
  - `main()` 调用 `eframe::run_native`
  - 窗口标题 "GGUF 推理引擎"
  - 初始窗口大小 1000x700
  - `native_options` 配置
- **验证**: `cargo run --features gui --bin gguf-gui` 启动窗口

---

## Phase 4: 功能视图

### T8: 对话视图
- **文件**: `src/gui/chat_view.rs`（新建）
- **改动**:
  - 对话区 ScrollArea，自动滚动到底部
  - 用户/助手气泡（不同背景色）
  - 流式 token 实时追加到当前助手气泡
  - 输入框（多行，Enter 发送，Shift+Enter 换行）
  - [发送] [重置上下文] [停止生成] 按钮
  - 生成中状态：输入框禁用、显示"生成中..."
  - 上下文超出提示
- **验证**: 多轮对话正常、流式输出、停止/重置功能

### T9: Prompt 补全视图
- **文件**: `src/gui/prompt_view.rs`（新建）
- **改动**:
  - Prompt 输入框（多行）
  - ☐ 流式输出 / ☐ 贪心解码
  - 输出区 ScrollArea
  - 统计行（耗时、tok/s、ctx）
  - [运行] [清除输出] [停止生成] 按钮
- **验证**: 单轮补全正常、流式/非流式切换、贪心覆盖

### T10: 模型信息视图
- **文件**: `src/gui/model_view.rs`（新建）
- **改动**:
  - 文件摘要面板（9 项信息）
  - KV 元数据表格 + 搜索过滤
  - 张量列表表格 + ☐ 显示全部
  - 词表查询（Token ID / Token 字符串）
  - [JSON 导出] [复制摘要] 按钮
- **验证**: 摘要正确、KV 搜索、张量展开、词表查询、JSON 导出

### T11: 设置面板
- **文件**: `src/gui/settings.rs`（新建）
- **改动**:
  - 温度 Slider (0.0–2.0)
  - Top-K DragValue (0–1000)
  - Top-P Slider (0.0–1.0)
  - Min-P Slider (0.0–1.0)
  - 重复惩罚 Slider (1.0–2.0)
  - 种子 TextEdit (u64)
  - 最大 Token DragValue (1–32768)
  - 贪心 Checkbox
  - 贪心勾选时禁用相关控件
  - 参数变更发送 `SamplerConfig` 到推理线程
- **验证**: 所有参数可调节、即时生效、贪心覆盖

---

## Phase 5: 集成测试

### T12: 端到端验证
- **改动**:
  - 无代码改动，纯验证
- **验证清单**:
  1. `cargo build --release --features gui` 编译成功
  2. `cargo build --release`（无 gui）编译成功
  3. `cargo test --release` 全部通过
  4. `cargo test --release --features gui` 全部通过
  5. GUI 启动 → 加载模型 → 对话 5 轮 → Prompt 补全 → 模型信息查看
  6. 停止生成 / 重置上下文 / 切换模型
  7. 设置面板参数调节即时生效
  8. JSON 导出正确

---

## 任务依赖图

```
T1 (依赖配置)
 ├── T2 (Engine 扩展)
 ├── T3 (Sampler 扩展)
 └── T4 (推理线程)
      ├── T5 (状态结构)
      └── T6 (主 App)
           ├── T7 (入口)
           ├── T8 (对话视图)
           ├── T9 (Prompt 视图)
           ├── T10 (模型信息视图)
           └── T11 (设置面板)
                └── T12 (端到端验证)
```

**可并行**：T2 和 T3 可并行；T8/T9/T10/T11 可并行（均依赖 T6）

**顺序关键路径**：T1 → T4 → T6 → T7 → T12

---

## 实施建议

1. **先 T1-T3**（基础 + 库 API 扩展），确保 `cargo check` 通过
2. **再 T4-T5**（推理线程），用命令行测试消息通道
3. **然后 T6-T7**（UI 框架），确保窗口可启动
4. **最后 T8-T11**（功能视图），逐个实现并测试
5. **T12 端到端验证**，对照 checklist.md 逐项检查
