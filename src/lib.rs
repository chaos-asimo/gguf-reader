//! # gguf — GGUF 元数据读取库
//!
//! 解析 GGUF（GGML Universal Format）文件的元数据：
//! - 文件头（magic / version / 张量数 / KV 数）
//! - 键值元数据（全部 13 种 `gguf_type`，含数组）
//! - 张量描述符（名称 / 形状 / 类型 / 数据偏移）
//!
//! 并扩展为**完整的 LLM 推理框架**：
//! - 量化反量化（Q4_0~Q8_K / F16 / BF16 / F32）
//! - 基础算子（GEMM / RMSNorm / Softmax / SiLU / RoPE）
//! - 模型 forward（llama / qwen2 / mistral）
//! - BPE 分词器
//! - 采样器（temperature / top-k / top-p / min-p / repeat penalty）
//! - KV-cache
//! - 推理引擎（prefill + decode + 流式生成）
//!
//! ## 快速开始
//!
//! ```no_run
//! use gguf::GgufFile;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file = GgufFile::from_path("model.gguf")?;
//! println!("architecture = {:?}", file.architecture());
//! println!("model name   = {:?}", file.model_name());
//! println!("tensors      = {}", file.header.n_tensors);
//! println!("kv pairs     = {}", file.header.n_kv);
//! # Ok(())
//! # }
//! ```

pub mod cursor;
pub mod error;
pub mod file;
pub mod header;
pub mod infer;
pub mod tensor;
pub mod types;

pub use crate::error::{GgufError, GgufResult};
pub use crate::file::GgufFile;
pub use crate::header::{GgufHeader, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
pub use crate::tensor::TensorInfo;
#[cfg(feature = "json")]
pub use crate::types::value_to_json;
pub use crate::types::{GgmlType, GgufArray, GgufType, GgufValue};
