//! 端到端推理集成测试。
//!
//! 构造一个最小的 2 层 llama GGUF（F32，含词表/BPE 分词器元数据），
//! 验证 [`gguf::GgufFile::from_reader`] → [`gguf::infer::Engine::generate`]
//! 全链路（分词 → 预填充 → 逐 token 解码 → KV cache → 采样）可运行。
//!
//! 另含 Q4_0 量化 roundtrip 测试（手工构造量化块 → 反量化 → 误差验证）。

use gguf::infer::quant;
use gguf::infer::sampler::SamplerConfig;
use gguf::infer::Engine;
use gguf::types::GgmlType;
use gguf::GgufFile;
use std::io::Cursor;

// ---------- GGUF 字节构造辅助 ----------

fn w_str(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

fn w_kv_scalar(buf: &mut Vec<u8>, key: &str, ty: i32, payload: &[u8]) {
    w_str(buf, key);
    buf.extend_from_slice(&ty.to_le_bytes());
    buf.extend_from_slice(payload);
}

fn w_kv_str(buf: &mut Vec<u8>, key: &str, s: &str) {
    w_str(buf, key);
    buf.extend_from_slice(&8i32.to_le_bytes()); // GGUF_TYPE_STRING
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

fn w_kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
    w_kv_scalar(buf, key, 4, &v.to_le_bytes()); // GGUF_TYPE_UINT32
}

fn w_kv_bool(buf: &mut Vec<u8>, key: &str, v: bool) {
    w_kv_scalar(buf, key, 7, &[if v { 1 } else { 0 }]); // GGUF_TYPE_BOOL
}

fn w_array_str(buf: &mut Vec<u8>, key: &str, items: &[&str]) {
    w_str(buf, key);
    buf.extend_from_slice(&9i32.to_le_bytes()); // GGUF_TYPE_ARRAY
    buf.extend_from_slice(&8i32.to_le_bytes()); // elem: STRING
    buf.extend_from_slice(&(items.len() as i64).to_le_bytes());
    for s in items {
        w_str(buf, s);
    }
}

fn w_tensor(buf: &mut Vec<u8>, name: &str, ne0: i64, ne1: i64, dtype: i32, offset: u64) {
    w_str(buf, name);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&ne0.to_le_bytes());
    buf.extend_from_slice(&ne1.to_le_bytes());
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// 张量布局：(name, rows, cols)，F32 行主序。
struct TensorSpec {
    name: String,
    rows: i64,
    cols: i64,
}

/// 构造最小 2 层 llama GGUF（F32）。
///
/// 维度：vocab=8, embed=4, n_heads=2, n_kv_heads=1, ffn=6, layers=2（head_dim=2）。
/// 分词器词表为单字符 token（a..h，id 0..7），无 BPE 合并规则。
/// `encode("ab")` → [BOS=0, "a"=0, "b"=1]。
///
/// `eos` 为 EOS token id：传 7（词表内）时贪心可能命中而提前停止；
/// 传 255（词表外）时贪心永不命中，可用于验证生成满 max_tokens。
fn build_min_llama(eos: u32) -> Vec<u8> {
    const P: &str = "llama";
    const N_LAYERS: usize = 2;
    const D: usize = 4; // embed
    const Q: usize = 2; // n_heads
    const KV: usize = 1; // n_kv_heads
    const F: usize = 6; // ffn
    const VOCAB: usize = 8;
    const HD: usize = D / Q; // 2

    // 按写入顺序生成张量列表
    let mut specs: Vec<TensorSpec> = Vec::new();
    specs.push(TensorSpec {
        name: format!("{P}.token_embd"),
        rows: VOCAB as i64,
        cols: D as i64,
    });
    for l in 0..N_LAYERS {
        specs.push(TensorSpec {
            name: format!("{P}.{l}.attn.wq"),
            rows: (Q * HD) as i64,
            cols: D as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.attn.wk"),
            rows: (KV * HD) as i64,
            cols: D as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.attn.wv"),
            rows: (KV * HD) as i64,
            cols: D as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.attn.wo"),
            rows: D as i64,
            cols: (Q * HD) as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.ffn.w1"),
            rows: F as i64,
            cols: D as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.ffn.w2"),
            rows: F as i64,
            cols: D as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.ffn.w3"),
            rows: D as i64,
            cols: F as i64,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.attn_norm"),
            rows: D as i64,
            cols: 1,
        });
        specs.push(TensorSpec {
            name: format!("{P}.{l}.ffn_norm"),
            rows: D as i64,
            cols: 1,
        });
    }
    specs.push(TensorSpec {
        name: format!("{P}.output_norm"),
        rows: D as i64,
        cols: 1,
    });
    specs.push(TensorSpec {
        name: format!("{P}.output"),
        rows: VOCAB as i64,
        cols: D as i64,
    });

    let n_kv: i64 = 14;
    let mut buf = Vec::new();
    // header
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(specs.len() as i64).to_le_bytes());
    buf.extend_from_slice(&n_kv.to_le_bytes());

    // KV 元数据（14 项）
    w_kv_str(&mut buf, "general.architecture", P);
    w_kv_u32(&mut buf, "general.alignment", 32);
    w_kv_u32(&mut buf, &format!("{P}.vocab_size"), VOCAB as u32);
    w_kv_u32(&mut buf, &format!("{P}.embedding_length"), D as u32);
    w_kv_u32(&mut buf, &format!("{P}.attention.head_count"), Q as u32);
    w_kv_u32(&mut buf, &format!("{P}.attention.head_count_kv"), KV as u32);
    w_kv_u32(&mut buf, &format!("{P}.ffn_length"), F as u32);
    w_kv_u32(&mut buf, &format!("{P}.block_count"), N_LAYERS as u32);
    w_kv_u32(&mut buf, &format!("{P}.tokenizer.n_vocab"), VOCAB as u32);
    w_array_str(
        &mut buf,
        &format!("{P}.tokenizer.tokens"),
        &["a", "b", "c", "d", "e", "f", "g", "h"],
    );
    w_kv_u32(&mut buf, &format!("{P}.tokenizer.n_merge"), 0);
    w_kv_u32(&mut buf, &format!("{P}.tokenizer.bos_token_id"), 0);
    w_kv_u32(&mut buf, &format!("{P}.tokenizer.eos_token_id"), eos);
    w_kv_bool(&mut buf, &format!("{P}.tokenizer.add_bos"), true);

    // 张量描述符 + 数据体
    let mut data: Vec<u8> = Vec::new();
    for spec in &specs {
        let bytes = (spec.rows * spec.cols * 4) as usize;
        let offset = data.len() as u64;
        w_tensor(&mut buf, &spec.name, spec.rows, spec.cols, 0, offset); // 0 = F32
                                                                         // 数据体：默认全零，部分张量置非零有限值
        data.extend(std::iter::repeat_n(0u8, bytes));
        if spec.name.contains("norm") {
            // RMSNorm 权重 = 1.0（恒等归一化）
            let one = 1.0f32.to_le_bytes();
            for (i, b) in data[offset as usize..].iter_mut().enumerate() {
                *b = one[i % 4];
            }
        } else if spec.name.ends_with(".ffn.w1")
            || spec.name.ends_with(".ffn.w3")
            || spec.name == format!("{P}.output")
        {
            // 稀疏非零权重（避免全零退化，保持有限）
            for i in (0..bytes).step_by(16) {
                data[i..i + 4].copy_from_slice(&0.5f32.to_le_bytes());
            }
        }
    }

    // 对齐填充（32 字节）
    let pad = (32 - (buf.len() as u64 % 32)) % 32;
    buf.extend(std::iter::repeat_n(0u8, pad as usize));
    buf.append(&mut data);

    buf
}

/// 贪心采样配置（确定性）。
fn greedy_cfg() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.0,
        repeat_penalty: 1.0,
        seed: 0,
    }
}

// ---------- 测试 ----------

/// 端到端：加载最小 GGUF → Engine → 生成 token，验证全链路可运行。
#[test]
fn end_to_end_generate() {
    let buf = build_min_llama(7);
    let file = GgufFile::from_reader(Cursor::new(buf.clone())).expect("GGUF 解析失败");
    assert_eq!(file.architecture().unwrap(), "llama");

    let mut engine = Engine::new(&file, greedy_cfg()).expect("Engine 构建失败");

    // 超参校验
    let hp = engine.hparams();
    assert_eq!(hp.n_layers, 2);
    assert_eq!(hp.vocab_size, 8);
    assert_eq!(hp.embed_dim, 4);

    // tokenize 一致性：BOS(0) + "a"(0) + "b"(1)
    assert_eq!(engine.tokenize("ab"), vec![0u32, 0, 1]);
    // detokenize 一致性
    assert_eq!(engine.detokenize(&[0, 0, 1]), "aab");

    let mut got: Vec<(u32, String)> = Vec::new();
    let text = engine
        .generate("ab", 4, |id, t| got.push((id, t.to_string())))
        .expect("generate 失败");

    // 生成 token 数 ≤ max_tokens（若首 token 即 EOS 则为 0，合法）
    assert!(got.len() <= 4, "生成数不应超过 max_tokens: {got:?}");
    for &(id, _) in &got {
        assert!(id < 8, "token id 应在词表内: {id}");
    }
    // 输出文本 = BOS(a) + prompt(2) + 生成 token（长度 = 3 + got.len()）
    assert_eq!(text.len(), 3 + got.len(), "BOS(1)+prompt(2)+生成: {text:?}");
    assert!(
        text.starts_with("aab"),
        "输出应为 BOS+prompt 开头: {text:?}"
    );

    // 贪心确定性：两次独立运行结果一致
    let file2 = GgufFile::from_reader(Cursor::new(buf)).unwrap();
    let mut engine2 = Engine::new(&file2, greedy_cfg()).unwrap();
    let text2 = engine2.complete("ab", 4).unwrap();
    assert_eq!(text, text2, "贪心采样应确定性一致");
}

/// 贪心生成恰好 max_tokens 个 token（EOS 设为词表外 255，永不命中）。
#[test]
fn generates_max_tokens() {
    let buf = build_min_llama(255);
    let file = GgufFile::from_reader(Cursor::new(buf)).unwrap();
    let mut engine = Engine::new(&file, greedy_cfg()).unwrap();

    let mut got = Vec::new();
    let text = engine
        .generate("ab", 5, |id, t| got.push((id, t.to_string())))
        .unwrap();

    assert_eq!(got.len(), 5, "应生成满 5 个 token: {got:?}");
    for &(id, _) in &got {
        assert!(id < 8, "token id 应在词表内: {id}");
    }
    // 输出 = BOS(a) + prompt(2) + 5 个生成 token（长度 8）
    assert_eq!(text.len(), 8, "BOS(1)+prompt(2)+生成(5): {text:?}");
    assert!(
        text.starts_with("aab"),
        "输出应为 BOS+prompt 开头: {text:?}"
    );
}

/// 生成 token 序列可复现（固定输入 + 贪心 → 固定输出）。
#[test]
fn greedy_reproducible() {
    let buf = build_min_llama(255);
    let file = GgufFile::from_reader(Cursor::new(buf)).unwrap();
    let mut e1 = Engine::new(&file, greedy_cfg()).unwrap();
    let mut e2 = Engine::new(&file, greedy_cfg()).unwrap();
    let t1 = e1.generate("a", 8, |_, _| {}).unwrap();
    let t2 = e2.generate("a", 8, |_, _| {}).unwrap();
    assert_eq!(t1, t2);
}

/// 量化 roundtrip：手工构造 Q4_0 块 → 反量化 → 与预期精确值比对。
#[test]
fn q4_0_roundtrip() {
    // Q4_0 块布局：d(f16, 2B) + 16 字节量化码（每字节 2 个 4-bit，值域 -8..=7）
    let d: f32 = 0.5;
    let d_bytes = quant::f32_to_f16(d);

    // 块 1：量化码全 0x88（两元素均为 0）→ 反量化全零
    let mut block1 = [0u8; 18];
    block1[0..2].copy_from_slice(&d_bytes);
    for b in block1[2..].iter_mut() {
        *b = 0x88;
    }
    let out1 = quant::dequantize(&block1, GgmlType::Q4_0, 32).expect("dequantize");
    assert_eq!(out1.len(), 32);
    assert!(out1.iter().all(|&v| v.is_finite() && v == 0.0f32));

    // 块 2：偶数字节 0x00（两元素 -8），奇数字节 0xFF（两元素 7）
    let mut block2 = [0u8; 18];
    block2[0..2].copy_from_slice(&d_bytes);
    for i in 0..16 {
        block2[2 + i] = if i % 2 == 0 { 0x00 } else { 0xFF };
    }
    let out2 = quant::dequantize(&block2, GgmlType::Q4_0, 32).expect("dequantize");
    for (i, v) in out2.iter().enumerate() {
        // 元素 i 来自字节 i/2：低 4 位（i 偶）或高 4 位（i 奇）
        let q = if i / 2 % 2 == 0 { -8.0f32 } else { 7.0f32 };
        let expected = d * q;
        assert!(
            (v - expected).abs() < 1e-5,
            "i={i}: got {} expected {}",
            v,
            expected
        );
    }

    // 多块：64 元素 = 2 个 Q4_0 块
    let mut multi = [0u8; 36];
    multi[0..18].copy_from_slice(&block1);
    multi[18..36].copy_from_slice(&block2);
    let out3 = quant::dequantize(&multi, GgmlType::Q4_0, 64).expect("dequantize");
    assert_eq!(out3.len(), 64);
    assert!(out3[..32].iter().all(|&v| v == 0.0f32));

    // 量化误差上界：Q4_0 最大量化步长 = d（1 个 4-bit 级距），误差 ≤ d/2
    // 对块 2 每个元素验证 |out - round-trip 真值| 由构造精确成立（此处仅验证有限性）
    assert!(out3.iter().all(|&v| v.is_finite()));
}

/// 缺失关键 KV（无 ffn_length）时 Engine 构建应返回错误（不 panic）。
#[test]
fn missing_kv_errors() {
    const P: &str = "llama";
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors = 0
    buf.extend_from_slice(&6i64.to_le_bytes()); // n_kv = 6
    w_kv_str(&mut buf, "general.architecture", P);
    w_kv_u32(&mut buf, "general.alignment", 32);
    w_kv_u32(&mut buf, &format!("{P}.vocab_size"), 8);
    w_kv_u32(&mut buf, &format!("{P}.embedding_length"), 4);
    w_kv_u32(&mut buf, &format!("{P}.attention.head_count"), 2);
    w_kv_u32(&mut buf, &format!("{P}.block_count"), 1);

    let file = GgufFile::from_reader(Cursor::new(buf)).unwrap();
    let r = Engine::new(&file, greedy_cfg());
    assert!(r.is_err(), "缺失 ffn_length 应报错");
}
