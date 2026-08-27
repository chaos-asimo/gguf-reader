//! 真实模型中文推理验证（需 491MB Qwen2.5-0.5B GGUF 文件）。
//!
//! 验证 GGUF 推理引擎端到端正确性：中文 prompt "你好" 贪心生成出**合法中文**
//! （无 U+FFFD 乱码、含 CJK 字符），且 token 序列确定性可复现。
//!
//! 历史：早期 attention 投影误用 matvec_colmajor（应为 matvec_colmajor_trans，
//! ggml 列主序 [in, out] 布局），导致 q/k/v/o 投影张量索引错乱，24 层累积后
//! logits 崩溃产生乱码。修正为 trans 后，中英文输出均恢复正常（英文 "Hello"
//! → "I am trying to create a simple program..."，中文 "你好" → 合法中文回答）。
//!
//! 默认 `cargo test` 跳过（需真实模型文件 + 较慢）；用以下命令显式运行：
//!
//! ```bash
//! cargo test --test chinese_inference -- --ignored
//! ```
//!
//! 需要 GGUF 文件位于 `target/qwen2.5-0.5b-instruct-q4_k_m-official.gguf`，
//! 或通过环境变量 `CHINESE_GGUF_PATH` 指定其他路径。

use gguf::infer::sampler::SamplerConfig;
use gguf::infer::Engine;
use gguf::GgufFile;
use std::path::PathBuf;

/// 贪心采样配置（确定性，便于复现）。
fn greedy_cfg() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        seed: 0,
    }
}

/// 定位 GGUF 文件：优先 `CHINESE_GGUF_PATH` 环境变量，否则回退到 target/ 默认路径。
fn locate_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHINESE_GGUF_PATH") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
        eprintln!("CHINESE_GGUF_PATH 指向的文件不存在: {}", pb.display());
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("qwen2.5-0.5b-instruct-q4_k_m-official.gguf");
    if fallback.exists() {
        Some(fallback)
    } else {
        None
    }
}

/// 检查字符串是否为"干净"的 UTF-8 文本：
/// - 无 U+FFFD（替换字符，乱码标志）
/// - 无控制字符（除 \t \n \r）
fn is_clean_utf8(s: &str) -> bool {
    !s.contains('\u{FFFD}') && !s.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r')
}

/// 统计 CJK 统一表意文字数（U+4E00..U+9FFF）。
fn cjk_count(s: &str) -> usize {
    s.chars().filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c)).count()
}

/// 端到端中文推理：加载真实 Qwen2.5 GGUF → 中文 prompt → 贪心生成 → 验证输出合法中文。
///
/// 断言：
/// 1. 生成 token 数 > 0
/// 2. 输出为干净 UTF-8（无 U+FFFD 替换字符、无控制字符）
/// 3. 输出含 CJK 统一表意文字（cjk_count >= 1）
/// 4. 贪心确定性：前 8 token 序列 == 已知正确序列
///    （attn 投影修正为 matvec_colmajor_trans 后的稳定序列；任一 forward
///     bug —— dequant / matvec / RoPE / softmax / attention / KV cache —— 都会偏离）
#[test]
#[ignore]
fn chinese_inference_no_garbage() {
    let model_path = locate_model().expect("找不到 Qwen2.5 GGUF 文件（需 target/ 下或设置 CHINESE_GGUF_PATH）");
    eprintln!("加载模型: {}", model_path.display());
    let file = GgufFile::from_path(&model_path).expect("GGUF 解析失败");
    let arch = file.architecture().unwrap_or("?");
    eprintln!("  架构={arch}  张量数={}", file.header.n_tensors);

    let mut engine = Engine::new(&file, greedy_cfg()).expect("Engine 构建失败");
    eprintln!(
        "  embed={} heads={} kv_heads={} ffn={} ctx={} vocab={}",
        engine.hparams().embed_dim,
        engine.hparams().n_heads,
        engine.hparams().n_kv_heads,
        engine.hparams().ffn_dim,
        engine.hparams().context_length,
        engine.hparams().vocab_size,
    );

    // 中文 prompt
    let prompt = "你好";
    let t0 = std::time::Instant::now();
    let mut got: Vec<(u32, String)> = Vec::new();
    let text = engine
        .generate(prompt, 32, |id, t| got.push((id, t.to_string())))
        .expect("generate 失败");
    let elapsed = t0.elapsed();
    eprintln!(
        "生成完成: {} tokens, 耗时 {:.2}s (prefill+generate 含加载)",
        got.len(),
        elapsed.as_secs_f64()
    );
    let ids: Vec<u32> = got.iter().map(|(id, _)| *id).collect();
    eprintln!("token ids: {ids:?}");
    eprintln!("prompt: {prompt}");
    eprintln!("输出:   {text}");
    eprintln!("CJK 字符数: {}", cjk_count(&text));
    eprintln!("是否干净 UTF-8: {}", is_clean_utf8(&text));

    // 1. 生成了 token
    assert!(!got.is_empty(), "未生成任何 token");
    // 2. 干净 UTF-8（无乱码标志 U+FFFD）
    assert!(
        is_clean_utf8(&text),
        "输出含 U+FFFD 或控制字符（乱码）: {text:?}"
    );
    // 3. 含 CJK 字符（中文输出）
    assert!(
        cjk_count(&text) >= 1,
        "输出不含任何 CJK 字符（应为中文回答）: {text:?}"
    );
    // 4. 贪心确定性：前 8 token 序列 == 已知正确序列（attn trans 修正后）
    let expected_prefix: [u32; 8] = [3837, 35946, 104133, 86119, 99172, 56007, 3837, 73670];
    let prefix: Vec<u32> = ids.iter().take(8).copied().collect();
    assert_eq!(
        prefix, expected_prefix.as_slice(),
        "前 8 token 偏离已知正确序列（forward 逻辑回归）:\n got: {prefix:?}\n exp: {expected_prefix:?}"
    );
    eprintln!("前 8 token 匹配已知正确序列 ✓（引擎 forward 正确）");
}
