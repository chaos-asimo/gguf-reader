//! 模型架构超参与 forward 实现。
//!
//! 各子模块负责一种 LLM 架构（llama / qwen2 / mistral）的：
//! - [`hparams`]：从 GGUF KV 元数据解析架构超参数
//! - forward：基于超参 + 张量做推理

pub mod hparams;
pub mod llama;
