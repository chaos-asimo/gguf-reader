//! GUI 状态与数据结构。

/// 应用状态
#[derive(Clone, Copy, PartialEq)]
pub enum AppState {
    /// 空闲，未加载模型
    Idle,
    /// 模型加载中
    Loading,
    /// 模型就绪
    Ready,
    /// 正在生成
    Generating,
    /// 错误
    Error,
}

/// 消息角色
#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn label(&self) -> &'static str {
        match self {
            Role::User => "用户",
            Role::Assistant => "助手",
        }
    }
}

/// 对话消息
#[derive(Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// 模型摘要信息
#[derive(Clone, Default)]
pub struct ModelSummary {
    pub name: String,
    pub arch: String,
    pub model_name: String,
    pub gguf_version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    pub file_size: u64,
    pub tensor_count: usize,
    pub kv_count: usize,
    pub load_ms: u128,
}

impl ModelSummary {
    pub fn size_mb(&self) -> f64 {
        self.file_size as f64 / (1024.0 * 1024.0)
    }
}

/// 生成统计
#[derive(Clone, Default)]
pub struct GenStats {
    pub elapsed_ms: u128,
    pub tokens: usize,
    pub ctx_len: usize,
    pub ctx_limit: usize,
}

impl GenStats {
    pub fn tok_per_s(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        self.tokens as f64 / (self.elapsed_ms as f64 / 1000.0)
    }
}
