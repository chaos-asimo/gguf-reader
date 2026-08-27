//! 架构超参数解析。
//!
//! 从 GGUF 元数据（`GgufFile`）读取 `general.architecture` 及对应的
//! `{arch}.*` KV，填充为 [`HParams`]。llama / qwen2 / mistral 共享同一套
//! 结构（仅个别键名/默认值不同）。

use crate::error::{GgufError, GgufResult};
use crate::file::GgufFile;
use crate::types::GgufValue;

/// 模型架构超参数。
///
/// 字段命名对齐 GGUF 键去掉 `{arch}.` 前缀后的名字（snake_case）。
#[derive(Debug, Clone, PartialEq)]
pub struct HParams {
    /// 架构名（general.architecture）。
    pub arch: String,
    /// 词表大小（`vocab_size`）。
    pub vocab_size: u32,
    /// 隐藏维度（`embedding_length`）。
    pub embed_dim: u32,
    /// 注意力头数（`attention.head_count`）。
    pub n_heads: u32,
    /// KV 头数（`attention.head_count_kv`），缺省等于 n_heads（MHA）。
    pub n_kv_heads: u32,
    /// 前馈中间维度（`ffn_length`）。
    pub ffn_dim: u32,
    /// 层数（`block_count`）。
    pub n_layers: u32,
    /// RoPE 基频（`attention.rope_freq_base`），缺省 10000.0。
    pub rope_freq_base: f64,
    /// RoPE 维度（`attention.head_count` 已含 head_dim = embed_dim / n_heads）。
    /// 部分模型用 `context_length`/`rope_freq_factor`，此处保留通用项。
    pub context_length: u32,
    /// 是否 pre-norm（`pre_norm` 缺省 true）。
    pub pre_norm: bool,
}

impl HParams {
    /// 每注意力头维度 = embed_dim / n_heads。
    pub fn head_dim(&self) -> u32 {
        self.embed_dim / self.n_heads.max(1)
    }
}

/// 从 GGUF 文件解析超参数（按 `general.architecture` 分派）。
pub fn parse(file: &GgufFile) -> GgufResult<HParams> {
    let arch = file
        .architecture()
        .ok_or_else(|| GgufError::InferenceError("missing general.architecture".into()))?;

    match arch {
        "llama" | "qwen2" | "mistral" => parse_generic(file, arch),
        // 架构名变体（qwen2.5 / qwen3 / llama3 等）使用相同 KV 键前缀
        a if a.starts_with("llama") => parse_generic(file, a),
        a if a.starts_with("qwen2") => parse_generic(file, a),
        a if a.starts_with("qwen3") => parse_generic(file, a),
        a if a.starts_with("mistral") => parse_generic(file, a),
        other => Err(GgufError::UnsupportedArchitecture(other.into())),
    }
}

/// 通用解析：读 `{arch}.{field}` KV。
fn parse_generic(file: &GgufFile, arch: &str) -> GgufResult<HParams> {
    let k = |f: &str| format!("{arch}.{f}");

    // vocab_size：优先 KV，缺失时从 token_embd 张量形状 [D, V] 推断
    // （兼容旧式 `{arch}.token_embd` 与新式无前缀 `token_embd.weight` 命名）
    let vocab_size = match u32_at(file, &k("vocab_size")) {
        Ok(v) => v,
        Err(_) => {
            let emb = file
                .find_tensor(&format!("{arch}.token_embd"))
                .or_else(|| file.find_tensor("token_embd.weight"))
                .ok_or_else(|| GgufError::InferenceError("vocab_size: KV 缺失且 token_embd 张量不存在".into()))?;
            if emb.shape.len() != 2 {
                return Err(GgufError::InferenceError(format!(
                    "vocab_size: token_embd shape 应为 [D, V]，实际 {:?}",
                    emb.shape
                )));
            }
            emb.shape[1] as u32
        }
    };
    let embed_dim = u32_at(file, &k("embedding_length"))?;
    let n_heads = u32_at(file, &k("attention.head_count"))?;
    // ffn_length 键名因模型而异：llama 用 ffn_length，qwen2 用 feed_forward_length
    let ffn_dim = u32_at(file, &k("ffn_length"))
        .or_else(|_| u32_at(file, &k("feed_forward_length")))?;
    let n_layers = u32_at(file, &k("block_count"))?;

    // KV 头数缺省等于 n_heads
    let n_kv_heads = file
        .get(&k("attention.head_count_kv"))
        .and_then(|v| v.as_i64())
        .map(|v| v as u32)
        .unwrap_or(n_heads);

    // RoPE 基频：llama 用 attention.rope_freq_base，qwen2 用 rope.freq_base，缺省 10000.0
    let rope_freq_base = file
        .get(&k("attention.rope_freq_base"))
        .and_then(|v| v.as_f64())
        .or_else(|| {
            file
                .get(&k("rope.freq_base"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(10000.0);

    // context_length 缺省 0（表示未指定，由引擎/调用方决定）
    // GGUF 通常存为 f32，用 as_f64 统一取整
    let context_length = file
        .get(&k("context_length"))
        .and_then(|v| v.as_f64())
        .map(|v| {
            if v.is_finite() && v >= 0.0 {
                v as u32
            } else {
                0
            }
        })
        .unwrap_or(0);

    // pre_norm 缺省 true
    let pre_norm = file
        .get(&k("pre_norm"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(HParams {
        arch: arch.to_string(),
        vocab_size,
        embed_dim,
        n_heads,
        n_kv_heads,
        ffn_dim,
        n_layers,
        rope_freq_base,
        context_length,
        pre_norm,
    })
}

/// 读取 `{key}` 为 u32（支持整数变体与 f32 整数值，非法/缺失返回错误）。
fn u32_at(file: &GgufFile, key: &str) -> GgufResult<u32> {
    let v = file
        .get(key)
        .ok_or_else(|| GgufError::InferenceError(format!("missing KV: {key}")))?;
    match v {
        GgufValue::U32(x) => Ok(*x),
        GgufValue::U16(x) => Ok(*x as u32),
        GgufValue::U8(x) => Ok(*x as u32),
        GgufValue::I32(x) if *x >= 0 => Ok(*x as u32),
        GgufValue::I64(x) if *x >= 0 => u32::try_from(*x)
            .map_err(|_| GgufError::InferenceError(format!("KV {key} out of u32 range: {x}"))),
        GgufValue::U64(x) => u32::try_from(*x)
            .map_err(|_| GgufError::InferenceError(format!("KV {key} out of u32 range: {x}"))),
        GgufValue::F32(x) => {
            // GGUF 可能将部分计数字段存为 f32；要求为 ≥0 的整数值
            if x.is_finite() && *x >= 0.0 && (x - x.round()).abs() < 1e-4 {
                Ok(*x as u32)
            } else {
                Err(GgufError::InferenceError(format!(
                    "KV {key} is not a non-negative integer: {x}"
                )))
            }
        }
        _ => Err(GgufError::InferenceError(format!(
            "KV {key} is not an integer: {:?}",
            v.value_type()
        ))),
    }
}

// ---------- 测试 ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor as IoCursor;

    /// GGUF 值封装（用于测试缓冲构造）。
    enum Kv {
        U32(u32),
        F32(f32),
        Str(&'static str),
        Bool(bool),
    }

    /// 构造 GGUF 缓冲：header + kvs（无张量）。
    fn build_buf(kvs: &[(&str, Kv)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&(kvs.len() as i64).to_le_bytes()); // n_kv
        for (key, val) in kvs {
            // key: u64 长度 + 字节
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            match val {
                Kv::U32(v) => {
                    buf.extend_from_slice(&4i32.to_le_bytes()); // UINT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(v) => {
                    buf.extend_from_slice(&6i32.to_le_bytes()); // FLOAT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(s) => {
                    buf.extend_from_slice(&8i32.to_le_bytes()); // STRING
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                Kv::Bool(b) => {
                    buf.extend_from_slice(&7i32.to_le_bytes()); // BOOL
                    buf.push(if *b { 1 } else { 0 });
                }
            }
        }
        buf
    }

    fn load(kvs: &[(&str, Kv)]) -> GgufFile {
        let buf = build_buf(kvs);
        GgufFile::from_reader(IoCursor::new(buf)).unwrap()
    }

    #[test]
    fn test_parse_llama() {
        let kvs: &[(&str, Kv)] = &[
            ("general.architecture", Kv::Str("llama")),
            ("llama.vocab_size", Kv::U32(32000)),
            ("llama.embedding_length", Kv::U32(4096)),
            ("llama.attention.head_count", Kv::U32(32)),
            ("llama.attention.head_count_kv", Kv::U32(8)),
            ("llama.ffn_length", Kv::U32(14336)),
            ("llama.block_count", Kv::U32(32)),
            ("llama.attention.rope_freq_base", Kv::F32(10000.0)),
            ("llama.context_length", Kv::F32(4096.0)),
            ("llama.pre_norm", Kv::Bool(true)),
        ];
        let f = load(kvs);
        let hp = parse(&f).unwrap();
        assert_eq!(hp.arch, "llama");
        assert_eq!(hp.vocab_size, 32000);
        assert_eq!(hp.embed_dim, 4096);
        assert_eq!(hp.n_heads, 32);
        assert_eq!(hp.n_kv_heads, 8);
        assert_eq!(hp.ffn_dim, 14336);
        assert_eq!(hp.n_layers, 32);
        assert!((hp.rope_freq_base - 10000.0).abs() < 1e-6);
        assert_eq!(hp.context_length, 4096);
        assert!(hp.pre_norm);
        assert_eq!(hp.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_missing_kv_errors() {
        // 只有 architecture，没有其它键
        let kvs: &[(&str, Kv)] = &[("general.architecture", Kv::Str("llama"))];
        let f = load(kvs);
        assert!(parse(&f).is_err());
    }

    #[test]
    fn test_unsupported_arch() {
        let kvs: &[(&str, Kv)] = &[("general.architecture", Kv::Str("bert"))];
        let f = load(kvs);
        assert!(matches!(
            parse(&f),
            Err(GgufError::UnsupportedArchitecture(_))
        ));
    }

    /// n_kv_heads / rope_freq_base / context_length 缺省回退。
    #[test]
    fn test_n_kv_heads_fallback() {
        let kvs: &[(&str, Kv)] = &[
            ("general.architecture", Kv::Str("mistral")),
            ("mistral.vocab_size", Kv::U32(32000)),
            ("mistral.embedding_length", Kv::U32(2560)),
            ("mistral.attention.head_count", Kv::U32(20)),
            ("mistral.ffn_length", Kv::U32(6912)),
            ("mistral.block_count", Kv::U32(32)),
        ];
        let f = load(kvs);
        let hp = parse(&f).unwrap();
        assert_eq!(hp.n_kv_heads, hp.n_heads); // 缺省回退
        assert_eq!(hp.rope_freq_base, 10000.0); // 缺省
        assert_eq!(hp.context_length, 0); // 缺省
    }
}
