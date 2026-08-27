//! LLaMA 架构 forward（llama / qwen2 / mistral 共享）。
//!
//! 张量命名遵循 GGUF 约定（`{prefix}.{i}.{...}`，`{prefix}` = llama/qwen2/mistral）。
//! 计算全程 f32；量化权重通过 [`quant::dequantize`] 物化为 f32 后调用 [`ops`]。

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{GgufError, GgufResult};
use crate::file::GgufFile;
use crate::infer::cache::Cache;
use crate::infer::ops;
use crate::infer::quant;
use crate::tensor::TensorInfo;
use crate::types::GgmlType;

use super::hparams::HParams;

/// 前缀（架构名），决定张量键。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefix {
    Llama,
    Qwen2,
    Mistral,
}

impl Prefix {
    pub fn as_str(&self) -> &'static str {
        match self {
            Prefix::Llama => "llama",
            Prefix::Qwen2 => "qwen2",
            Prefix::Mistral => "mistral",
        }
    }
    /// 从 HParams 推断前缀（支持架构名变体）。
    pub fn from_arch(arch: &str) -> GgufResult<Prefix> {
        if arch.starts_with("llama") {
            Ok(Prefix::Llama)
        } else if arch.starts_with("qwen2") {
            Ok(Prefix::Qwen2)
        } else if arch.starts_with("qwen3") {
            // qwen3 张量命名同 qwen2
            Ok(Prefix::Qwen2)
        } else if arch.starts_with("mistral") {
            Ok(Prefix::Mistral)
        } else {
            Err(GgufError::UnsupportedArchitecture(arch.into()))
        }
    }
}

/// 一个 Transformer 层的权重名字（owned，不借用 self，避免 E0502）。
///
/// 仅持有张量名，实际数据通过 [`LlamaModel::materialize`] 按名取用。
#[derive(Debug, Clone)]
struct LayerWeights {
    wq: String,
    wk: String,
    wv: String,
    wo: String,
    b_attn: Option<String>,
    b_k: Option<String>,
    b_v: Option<String>,
    w_ffn_up: String,
    w_ffn_gate: String,
    w_ffn_down: String,
    norm_attn: String,
    norm_ffn: String,
}

/// 模型句柄：持有 GGUF 文件、超参、前缀与 KV cache。
pub struct LlamaModel<'a> {
    file: &'a GgufFile,
    hp: HParams,
    prefix: Prefix,
    /// 每层张量名映射（支持新旧两种命名风格）。
    layer_names: Vec<LayerNames>,
    /// 顶层张量名映射。
    top_names: TopNames,
    cache: Option<Cache>,
    #[allow(dead_code)]
    rope_base: f64,
    inv_freq: Vec<f32>,
    /// 物化的 f32 权重缓存（张量名 → 共享 flat vec）。
    ///
    /// 首次访问时反量化并缓存，后续 token 通过 `Arc::clone`（原子引用计数自增）复用，
    /// 避免每 token 全量 dequantize，也避免克隆大张量（如 token_embd 93MB）的内存开销。
    weight_cache: HashMap<String, Arc<Vec<f32>>>,
}

/// 每层张量名（已解析为实际 GGUF 张量名）。
#[derive(Debug, Clone)]
struct LayerNames {
    wq: String,
    wk: String,
    wv: String,
    wo: String,
    b_q: Option<String>,
    b_k: Option<String>,
    b_v: Option<String>,
    w_ffn_up: String,
    w_ffn_gate: String,
    w_ffn_down: String,
    norm_attn: String,
    norm_ffn: String,
}

/// 顶层张量名。
#[derive(Debug, Clone)]
struct TopNames {
    token_embd: String,
    output: String,
    output_norm: String,
}

impl<'a> LlamaModel<'a> {
    /// 从 GGUF 文件构建模型（解析超参、校验关键张量存在）。
    pub fn new(file: &'a GgufFile) -> GgufResult<Self> {
        let hp = super::hparams::parse(file)?;
        let prefix = Prefix::from_arch(&hp.arch)?;
        let n_kv = hp.n_kv_heads as usize;
        let cache = Some(Cache::new(
            hp.n_layers as usize,
            n_kv,
            hp.head_dim() as usize,
        ));
        // 逆频率表（RoPE）
        let head_dim = hp.head_dim() as usize;
        let inv_freq = build_inv_freq(head_dim, hp.rope_freq_base);
        let rope_base = hp.rope_freq_base;

        // 解析张量名映射（新旧命名风格）
        // 旧式: {arch}.token_embd ；新式（无前缀）: token_embd.weight
        let p = &hp.arch;
        let top_names = TopNames {
            token_embd: resolve_tensor(file, &format!("{p}.token_embd"), "token_embd.weight")?,
            output: resolve_tensor(file, &format!("{p}.output"), "output.weight")?,
            output_norm: resolve_tensor(file, &format!("{p}.output_norm"), "output_norm.weight")?,
        };
        let mut layer_names = Vec::with_capacity(hp.n_layers as usize);
        for l in 0..hp.n_layers as usize {
            let ls = l.to_string();
            layer_names.push(LayerNames {
                wq: resolve_tensor(file, &format!("{p}.{ls}.attn.wq"), &format!("blk.{ls}.attn_q.weight"))?,
                wk: resolve_tensor(file, &format!("{p}.{ls}.attn.wk"), &format!("blk.{ls}.attn_k.weight"))?,
                wv: resolve_tensor(file, &format!("{p}.{ls}.attn.wv"), &format!("blk.{ls}.attn_v.weight"))?,
                wo: resolve_tensor(file, &format!("{p}.{ls}.attn.wo"), &format!("blk.{ls}.attn_output.weight"))?,
                b_q: find_opt(file, &format!("{p}.{ls}.attn.b_q"), &format!("blk.{ls}.attn_q.bias")),
                b_k: find_opt(file, &format!("{p}.{ls}.attn.b_k"), &format!("blk.{ls}.attn_k.bias")),
                b_v: find_opt(file, &format!("{p}.{ls}.attn.b_v"), &format!("blk.{ls}.attn_v.bias")),
                w_ffn_up: resolve_tensor(file, &format!("{p}.{ls}.ffn.w1"), &format!("blk.{ls}.ffn_up.weight"))?,
                w_ffn_gate: resolve_tensor(file, &format!("{p}.{ls}.ffn.w2"), &format!("blk.{ls}.ffn_gate.weight"))?,
                w_ffn_down: resolve_tensor(file, &format!("{p}.{ls}.ffn.w3"), &format!("blk.{ls}.ffn_down.weight"))?,
                norm_attn: resolve_tensor(file, &format!("{p}.{ls}.attn_norm"), &format!("blk.{ls}.attn_norm.weight"))?,
                norm_ffn: resolve_tensor(file, &format!("{p}.{ls}.ffn_norm"), &format!("blk.{ls}.ffn_norm.weight"))?,
            });
        }

        Ok(Self {
            file,
            hp,
            prefix,
            layer_names,
            top_names,
            cache,
            rope_base,
            inv_freq,
            weight_cache: HashMap::new(),
        })
    }

    /// 超参只读访问。
    pub fn hparams(&self) -> &HParams {
        &self.hp
    }

    /// 物化（反量化）指定张名为 f32 flat vec，首次调用时缓存。
    ///
    /// 返回 `Arc<Vec<f32>>`（共享引用计数）：首次调用反量化并插入缓存，
    /// 后续调用仅 `Arc::clone`（原子自增），无内存拷贝。
    ///
    /// 调用方须确保 `name` 不借用 `self`（先 clone 到局部变量），否则 E0502。
    fn materialize(&mut self, name: &str) -> GgufResult<Arc<Vec<f32>>> {
        if let Some(w) = self.weight_cache.get(name) {
            return Ok(Arc::clone(w));
        }
        let t = self.t(name)?;
        let span = self.file.tensor_physical_span(t)?;
        let n = t.num_elements();
        let w = dequant_tensor_physical(self.file, t, span)?;
        // 防御性检查：非量化 dtype 的张量（bias/norm 等 F32）应返回与 num_elements 相同长度的数据。
        // 若长度不一致，说明 tensor_data 因 header size 与物理布局不一致被截断，
        // 静默使用截断数据会导致后续注入错误（如 bias 错位）。
        if w.len() != n as usize {
            return Err(GgufError::InferenceError(format!(
                "materialize({name}): dequantized length {} != num_elements {} \
                 (dtype={:?}, span={})；header size 与物理布局不一致，数据可能被截断",
                w.len(), n, t.dtype, span
            )));
        }
        let arc = Arc::new(w);
        self.weight_cache.insert(name.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// 前缀只读访问。
    pub fn prefix(&self) -> Prefix {
        self.prefix
    }

    /// 前向传播：输入一批 token id（同位置），输出 `vocab_size` 的 logits。
    ///
    /// `cache` 为 `None` 时不保存 KV（用于预填充或一次性推理）。
    pub fn forward(&mut self, tokens: &[u32], positions: &[i64]) -> GgufResult<Vec<f32>> {
        self.forward_internal(tokens, positions, None)
    }

    /// 带 KV cache 的前向传播（增量模式，供多轮对话使用）。
    ///
    /// 调用前须确保 `cache_len()` 与 `positions` 起点一致（positions[0] == cache_len），
    /// 即本方法只在 cache 已持有全部历史 K/V 时追加新 token。
    pub fn forward_cached(&mut self, tokens: &[u32], positions: &[i64]) -> GgufResult<Vec<f32>> {
        self.forward_internal(tokens, positions, None)
    }

    /// 当前 KV cache 已持有的 token 数（各层一致）。
    pub fn cache_len(&self) -> usize {
        self.cache
            .as_ref()
            .and_then(|c| c.layer(0))
            .map_or(0, |l| l.seq_len())
    }

    /// 清空 KV cache（开始新对话）。
    pub fn reset_cache(&mut self) {
        if let Some(c) = self.cache.as_mut() {
            c.clear_all();
        }
    }

    /// 前向传播并 dump 最后一个 token 的 final-norm 隐藏态（用于与参考实现对照）。
    ///
    /// `hidden_path` 为二进制 f32 输出路径（embed_dim 个元素）。
    pub fn forward_dump_hidden(
        &mut self,
        tokens: &[u32],
        positions: &[i64],
        hidden_path: &str,
    ) -> GgufResult<(Vec<f32>, Vec<f32>)> {
        let logits = self.forward_internal(tokens, positions, Some(hidden_path))?;
        let d = self.hp.embed_dim as usize;
        let n = tokens.len();
        // hidden 已由 forward_internal 在 dump 时计算；这里重新取 final norm 结果不现实，
        // 改为：forward_internal 把 hidden 也写盘（路径 + ".hidden"），读取之。
        let raw = std::fs::read(format!("{hidden_path}.hidden")).map_err(|e| {
            GgufError::InferenceError(format!("读取 dump 的 hidden 失败 {hidden_path}.hidden: {e}"))
        })?;
        let hidden = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        let _ = (d, n);
        Ok((logits, hidden))
    }

    fn forward_internal(
        &mut self,
        tokens: &[u32],
        positions: &[i64],
        hidden_dump: Option<&str>,
    ) -> GgufResult<Vec<f32>> {
        assert_eq!(
            tokens.len(),
            positions.len(),
            "tokens 与 positions 长度必须一致"
        );
        let n = tokens.len();
        if n == 0 {
            return Err(GgufError::InferenceError("empty input".into()));
        }

        let d0 = self.hp.embed_dim as usize;
        let mut hidden = self.embed(tokens)?; // [n, D]
        let layer = self.hp.n_layers as usize;
        let inv = 1.0f32 / (self.hp.head_dim() as f32).sqrt();

        // 逐层 dump：每层开始前记录最后一个 token 的 residual 状态到 {base}_layer_{l}.bin
        for l in 0..layer {
            if let Some(base) = hidden_dump {
                dump_layer_state(base, l, &hidden[(n - 1) * d0..n * d0]);
            }
            let lw = self.layer_weights(l)?;
            let d0 = self.hp.embed_dim as usize;
            // 1. 保留原始 hidden（残差连接用）
            let mut residual = hidden.clone();
            // 2. pre-norm（attention）
            let attn_in = self.apply_rmsnorm(&residual, &lw.norm_attn, n)?;
            // 3. 注意力（输出 = wo 投影）
            let attn_out = self.attention_block(&attn_in, &lw, positions, l, n, inv)?;
            if attn_out.iter().any(|v| v.is_nan()) {
                if cfg!(debug_assertions) {
                    let cnt = attn_out.iter().filter(|v| v.is_nan()).count();
                    let first = attn_out.iter().position(|v| v.is_nan());
                    eprintln!("[debug] layer {l} attention output: {cnt} NaNs, first at {first:?}");
                }
                return Err(GgufError::InferenceError(format!("NaN after layer {l} attention")));
            }
            // 4. 残差：原始 hidden + attention 输出
            for i in 0..n * d0 {
                residual[i] += attn_out[i];
            }
            hidden = residual;
            // 5. pre-norm（ffn）
            let ffn_in = self.apply_rmsnorm(&hidden, &lw.norm_ffn, n)?;
            // 6. FFN
            let ffn_out = self.ffn_block(&ffn_in, &lw, n)?;
            if ffn_out.iter().any(|v| v.is_nan()) {
                if cfg!(debug_assertions) {
                    let cnt = ffn_out.iter().filter(|v| v.is_nan()).count();
                    let first = ffn_out.iter().position(|v| v.is_nan());
                    eprintln!("[debug] layer {l} ffn output: {cnt} NaNs, first at {first:?}");
                }
                return Err(GgufError::InferenceError(format!("NaN after layer {l} ffn")));
            }
            // 7. 残差：ffn 前的 hidden + FFN 输出
            for i in 0..n * d0 {
                hidden[i] += ffn_out[i];
            }
        }

        // 7. final norm
        let output_norm_name = self.top_names.output_norm.clone();
        let normed = self.apply_rmsnorm(&hidden, &output_norm_name, n)?;
        if let Some(path) = hidden_dump {
            let d = self.hp.embed_dim as usize;
            let last = &normed[(n - 1) * d..n * d];
            // 以二进制 f32 写出（4B/元素），Python 侧 np.fromfile 读取；
            // 同时写一份 "{path}.hidden" 供 forward_dump_hidden 读回。
            let bytes: Vec<u8> = last.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, &bytes).map_err(|e| {
                GgufError::InferenceError(format!("dump hidden 失败 {path}: {e}"))
            })?;
            std::fs::write(format!("{path}.hidden"), bytes).map_err(|e| {
                GgufError::InferenceError(format!("dump hidden 失败 {path}.hidden: {e}"))
            })?;
        }
        // 8. lm_head
        let logits = self.lm_head(&normed, n)?;
        Ok(logits)
    }

    // ---------- 内部步骤 ----------

    fn t(&self, name: &str) -> GgufResult<&'a TensorInfo> {
        self.file
            .find_tensor(name)
            .ok_or_else(|| GgufError::MissingTensor {
                name: name.to_string(),
                kind: "tensor",
            })
    }

    /// 嵌入：返回 [n, D] 的 hidden（行主序）。
    ///
    /// GGUF 张量 shape=[embed_dim, vocab_size]，dim[0] 连续（列主序）。
    /// 反量化后 w 为 flat vec，token t 的嵌入位于 w[t*embed_dim..(t+1)*embed_dim]。
    fn embed(&mut self, tokens: &[u32]) -> GgufResult<Vec<f32>> {
        let name = self.top_names.token_embd.clone();
        let w = self.materialize(&name)?;
        let d = self.hp.embed_dim as usize;
        let mut out = vec![0f32; tokens.len() * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let base = tok as usize * d;
            if base + d > w.len() {
                return Err(GgufError::InferenceError(format!(
                    "embed out of bounds: tok={tok}, base={base}, d={d}, w.len()={}",
                    w.len()
                )));
            }
            out[i * d..(i + 1) * d].copy_from_slice(&w[base..base + d]);
        }
        Ok(out)
    }

    fn apply_rmsnorm(&mut self, x: &[f32], name: &str, n: usize) -> GgufResult<Vec<f32>> {
        let weight = self.materialize(name)?;
        let d = weight.len();
        let mut out = x.to_vec();
        for i in 0..n {
            ops::rmsnorm(&mut out[i * d..(i + 1) * d], &weight, 1e-6f32);
        }
        Ok(out)
    }

    fn layer_weights(&self, l: usize) -> GgufResult<LayerWeights> {
        let ln = &self.layer_names[l];
        // 校验所有必需张量存在（返回名字即可，数据由 materialize 按名取用）
        for name in [&ln.wq, &ln.wk, &ln.wv, &ln.wo, &ln.w_ffn_up, &ln.w_ffn_gate, &ln.w_ffn_down, &ln.norm_attn, &ln.norm_ffn] {
            self.t(name)?;
        }
        Ok(LayerWeights {
            wq: ln.wq.clone(),
            wk: ln.wk.clone(),
            wv: ln.wv.clone(),
            wo: ln.wo.clone(),
            b_attn: ln.b_q.clone(),
            b_k: ln.b_k.clone(),
            b_v: ln.b_v.clone(),
            w_ffn_up: ln.w_ffn_up.clone(),
            w_ffn_gate: ln.w_ffn_gate.clone(),
            w_ffn_down: ln.w_ffn_down.clone(),
            norm_attn: ln.norm_attn.clone(),
            norm_ffn: ln.norm_ffn.clone(),
        })
    }

    fn attention_block(
        &mut self,
        x: &[f32],
        lw: &LayerWeights,
        positions: &[i64],
        l: usize,
        n: usize,
        inv: f32,
    ) -> GgufResult<Vec<f32>> {
        let d = self.hp.embed_dim as usize;
        let q = self.hp.n_heads as usize;
        let kv = self.hp.n_kv_heads as usize;
        let hd = self.hp.head_dim() as usize;

        let _ = (q, hd, d, kv); // 避免未使用变量警告
        // Qwen2 确有 attn bias（GGUF bias 与 HF 完全一致，cos=1.0，非损坏），必须注入。
        // 之前误判 bias 损坏而跳过，导致 x 量级失控、输出乱码。
        let b_attn = lw.b_attn.as_ref().map(|n| self.materialize(n)).transpose()?.unwrap_or_default();
        let b_k = lw.b_k.as_ref().map(|n| self.materialize(n)).transpose()?.unwrap_or_default();
        let b_v = lw.b_v.as_ref().map(|n| self.materialize(n)).transpose()?.unwrap_or_default();
        // QKV / O 投影权重
        let wq = self.materialize(&lw.wq)?;
        let wk = self.materialize(&lw.wk)?;
        let wv = self.materialize(&lw.wv)?;
        let wo = self.materialize(&lw.wo)?;

        // 每 token 计算 Q/K/V 并做 GQA 注意力
        let mut out = vec![0f32; n * d];
        let inv_freq = self.inv_freq.clone();
        for (tok, pos) in positions.iter().enumerate() {
            let pos = *pos;
            let base = tok * d;
            let xb = &x[base..base + d];
            // Q: ggml ne=[d, q*hd]（in=d 连续，out=q*hd 外层）。qv[i∈q*hd] = Σ_j a[j + i*d]*x[j]
            let mut qv = vec![0f32; q * hd];
            ops::matvec_colmajor_trans(&wq, d as u64, (q * hd) as u64, xb, &mut qv, None)?;
            for (i, b) in b_attn.iter().enumerate().take(q * hd) {
                qv[i] += b;
            }
            // RoPE：qv 布局为 [q 个 head × hd]，rope 把每 head 当作一个"token 位置"，
            // 因此 positions 长度须 = q，且每 head 用同一 pos（否则 head 1..q-1 退化为 pos=0）。
            ops::rope(&mut qv, &vec![pos; q], &inv_freq);
            // K/V: ggml ne=[d, kv*hd]（in=d 连续，out=kv*hd 外层）
            let mut kbuf = vec![0f32; kv * hd];
            ops::matvec_colmajor_trans(&wk, d as u64, (kv * hd) as u64, xb, &mut kbuf, None)?;
            for (i, b) in b_k.iter().enumerate().take(kv * hd) {
                kbuf[i] += b;
            }
            let mut vbuf = vec![0f32; kv * hd];
            ops::matvec_colmajor_trans(&wv, d as u64, (kv * hd) as u64, xb, &mut vbuf, None)?;
            for (i, b) in b_v.iter().enumerate().take(kv * hd) {
                vbuf[i] += b;
            }
            // K RoPE：kbuf 布局 [kv 个 head × hd]，positions 长度 = kv，每 head 用同一 pos
            ops::rope(&mut kbuf, &vec![pos; kv], &inv_freq);
            // 追加到 cache
            if let Some(c) = self.cache.as_mut().and_then(|c| c.layer_mut(l)) {
                c.append(&kbuf, &vbuf);
            }
            // 注意力：取 cache 前缀（含本 token）
            let (seq, k_all, v_all) = if let Some(cl) = self.cache.as_ref().and_then(|c| c.layer(l))
            {
                (cl.seq_len(), cl.get_k().to_vec(), cl.get_v().to_vec())
            } else {
                (1, kbuf.clone(), vbuf.clone())
            };
            let o = self.attend(&qv, &k_all, &v_all, seq, q, kv, hd, inv, pos)?;
            // O 投影: ggml ne=[q*hd, d]（in=q*hd 连续，out=d 外层）。out[i∈d] = Σ_j a[j + i*(q*hd)]*x[j]
            ops::matvec_colmajor_trans(
                &wo,
                (q * hd) as u64,
                d as u64,
                &o,
                &mut out[base..base + d],
                None,
            )?;
        }
        Ok(out)
    }

    /// GQA 注意力：Q [q*hd]，K/V [seq*kv*hd] → O [q*hd]。
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        qn: usize,
        kvn: usize,
        hd: usize,
        inv: f32,
        pos: i64,
    ) -> GgufResult<Vec<f32>> {
        let mut o = vec![0f32; qn * hd];
        let qpk = qn / kvn; // 每个 kv-head 服务的 q-head 数（grouped 映射：连续 qpk 个 q-head 共享一个 kv-head）
        for h in 0..qn {
            let khead = h / qpk; // GQA 映射（LLaMA/Qwen2 约定：h / (n_heads / n_kv_heads)）
            let qh = &q[h * hd..(h + 1) * hd];
            let mut scores = vec![0f32; seq];
            for s in 0..seq {
                let kb = &k[(s * kvn + khead) * hd..(s * kvn + khead) * hd + hd];
                let mut dot = 0f32;
                for i in 0..hd {
                    dot += qh[i] * kb[i];
                }
                // 因果掩码：基于绝对位置 pos。cache 按 [0, seq) 绝对位置索引，
                // 本 token 只能看到位置 <= pos 的 token（decode 时 tok=0 但 pos=len-1）。
                if (s as i64) <= pos {
                    scores[s] = dot * inv;
                } else {
                    scores[s] = f32::NEG_INFINITY;
                }
            }
            ops::softmax(&mut scores);
            let mut oh = vec![0f32; hd];
            for s in 0..seq {
                let vb = &v[(s * kvn + khead) * hd..(s * kvn + khead) * hd + hd];
                let w = scores[s];
                if w.abs() > 1e-9 {
                    for i in 0..hd {
                        oh[i] += w * vb[i];
                    }
                }
            }
            o[h * hd..(h + 1) * hd].copy_from_slice(&oh);
        }
        Ok(o)
    }

    fn ffn_block(&mut self, x: &[f32], lw: &LayerWeights, n: usize) -> GgufResult<Vec<f32>> {
        let d = self.hp.embed_dim as usize;
        let f = self.hp.ffn_dim as usize;
        let w1 = self.materialize(&lw.w_ffn_up)?;
        let w2 = self.materialize(&lw.w_ffn_gate)?;
        let w3 = self.materialize(&lw.w_ffn_down)?;
        let mut out = vec![0f32; n * d];
        for tok in 0..n {
            let base = tok * d;
            let xb = &x[base..base + d];
            // w1 (up): ggml ne=[d, f]（in=d 连续，out=f 外层）。up[i∈f] = Σ_j a[j + i*d]*x[j]。
            // matvec_colmajor_trans(dim0=d, dim1=f)：y[i] = Σ_j a[j + i*d]*x[j] ✓
            let mut up = vec![0f32; f];
            ops::matvec_colmajor_trans(&w1, d as u64, f as u64, xb, &mut up, None)?;
            // w2 (gate): 同 w1
            let mut gate = vec![0f32; f];
            ops::matvec_colmajor_trans(&w2, d as u64, f as u64, xb, &mut gate, None)?;
            ops::silu(&mut gate);
            for i in 0..f {
                gate[i] *= up[i];
            }
            // w3 (down): ggml ne=[f, d]（in=f 连续，out=d 外层）。out[i∈d] = Σ_j a[j + i*f]*x[j]。
            // matvec_colmajor_trans(dim0=f, dim1=d)
            ops::matvec_colmajor_trans(
                &w3,
                f as u64,
                d as u64,
                &gate,
                &mut out[base..base + d],
                None,
            )?;
        }
        Ok(out)
    }

    fn lm_head(&mut self, x: &[f32], n: usize) -> GgufResult<Vec<f32>> {
        let d = self.hp.embed_dim as usize;
        let output_name = self.top_names.output.clone();
        let w = self.materialize(&output_name)?;
        // 返回最后一个 token 的 logits。
        // output 列主序 shape=[d, vocab_size]：dim0=d 连续，flat = i + v*d（i∈[0,d), v∈[0,vocab)）。
        // logits[v] = Σ_i output[i, v] * last[i] = Σ_i w[i + v*d] * last[i]。
        let last = x[(n - 1) * d..n * d].to_vec();
        let vocab = self.hp.vocab_size as usize;
        let mut logits = vec![0f32; vocab];
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            logits.par_iter_mut().enumerate().for_each(|(v, lv)| {
                let base = v * d;
                let mut acc = 0f32;
                for i in 0..d {
                    acc += w[base + i] * last[i];
                }
                *lv = acc;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for v in 0..vocab {
                let base = v * d;
                let mut acc = 0f32;
                for i in 0..d {
                    acc += w[base + i] * last[i];
                }
                logits[v] = acc;
            }
        }
        Ok(logits)
    }
}

/// 按首选名称查找张量，找不到则尝试备用名称；都找不到则报错。
/// 返回实际匹配的张量名（用于后续 `t()` 查找）。
fn resolve_tensor(file: &GgufFile, primary: &str, fallback: &str) -> GgufResult<String> {
    if file.find_tensor(primary).is_some() {
        Ok(primary.to_string())
    } else if file.find_tensor(fallback).is_some() {
        Ok(fallback.to_string())
    } else {
        Err(GgufError::MissingTensor {
            name: primary.to_string(),
            kind: "tensor",
        })
    }
}

/// 可选查找张量（两个名称都不存在时返回 None）。
/// 返回实际匹配的张量名。
fn find_opt(file: &GgufFile, primary: &str, fallback: &str) -> Option<String> {
    if file.find_tensor(primary).is_some() {
        Some(primary.to_string())
    } else if file.find_tensor(fallback).is_some() {
        Some(fallback.to_string())
    } else {
        None
    }
}

/// 物理格式推断：按"每 256 元素字节数"在已知标准布局中查找匹配 dtype。
///
/// 某些 GGUF 文件（如 Qwen2.5-0.5B Q4_K_M）存在 dtype 标记与物理量化格式
/// 系统性错位：
/// - ffn_down 标记 Q4_K 物理占用 176B/256elem（= Q5_K 比特率）
/// - ffn_down 标记 Q3_K_M 物理占用 144B/256elem（= Q4_K 比特率）
/// - attn_v / output 标记 Q5_1 物理占用 210B/256elem（= Q6_K 比特率）
///
/// 此函数按物理字节还原真实存储格式，支持所有量化类型（Q4_0~Q8_K），
/// 避免反量化读错 block 边界产生乱码。
fn infer_k_quant_dtype(span: u64, n: u64, marked: GgmlType) -> Option<GgmlType> {
    // 标记 dtype 必须是量化类型（F32/F16/BF16 无 block 结构，不参与推断）
    if !matches!(
        marked,
        GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K_S
            | GgmlType::Q3_K_M
            | GgmlType::Q3_K_L
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
    ) {
        return None;
    }
    if n == 0 {
        return None;
    }
    // 统一换算到 256 元素粒度：
    // - 32-elem block 的 dtype（Q4_0/Q5_0/Q5_1/Q8_0 等）：每 256 elem = 8 个 32-elem block
    // - 256-elem block 的 dtype（K-quant）：直接对比
    let marked_bs = marked.block_size()?;
    let blocks = if marked_bs == 32 { n / 32 } else { n / 256 };
    if blocks == 0 {
        return None;
    }
    // bytes per 256-element group（32-elem block 时 ×8 换算）
    let bytes_per_256 = if marked_bs == 32 {
        span * 8 / blocks
    } else {
        span / blocks
    };
    let remainder = span % blocks;
    if remainder > 256 {
        return None; // padding 超过一个 block，数据异常
    }
    let table: &[(u64, GgmlType)] = &[
        (84, GgmlType::Q2_K),
        (110, GgmlType::Q3_K_S),
        (114, GgmlType::Q3_K_M),
        (144, GgmlType::Q4_K),
        (176, GgmlType::Q5_K),
        (210, GgmlType::Q6_K),
        (292, GgmlType::Q8_K),
    ];
    table
        .iter()
        .find(|(b, _)| *b == bytes_per_256)
        .map(|(_, dt)| *dt)
}

/// 读取张量物理字节并按真实存储格式反量化为 f32。
///
/// `span` 为张量物理字节跨度（[`GgufFile::tensor_physical_span`]），`n` 为元素总数。
/// - 标记 dtype 与物理 span 不一致时，按物理字节推断真实 dtype（见
///   [`infer_k_quant_dtype`]），支持所有量化类型（32-elem 和 256-elem block）。
/// - 推断成功时按物理跨度取数（`use_physical=true`），截取精确字节去掉尾部 padding。
/// - 推断失败时回退到标记 dtype（数据长度以 header size 为准）。
pub fn dequant_tensor_physical(
    file: &GgufFile,
    t: &TensorInfo,
    span: u64,
) -> GgufResult<Vec<f32>> {
    let n = t.num_elements();
    let marked_bb = t
        .dtype
        .block_bytes()
        .ok_or_else(|| GgufError::DequantError {
            dtype: format!("{:?}", t.dtype),
            expected: 0,
            actual: 0,
        })?;
    let expected_span = n / t.dtype.block_size().unwrap_or(1) * marked_bb;

    let (actual_dtype, use_physical) = if span != expected_span {
        match infer_k_quant_dtype(span, n, t.dtype) {
            Some(dt) => (dt, true), // 推断出真实物理格式，按物理跨度取数
            None => (t.dtype, false), // 无法推断：按标记 dtype + header size 取数
        }
    } else {
        (t.dtype, false)
    };

    let dd = actual_dtype
        .block_bytes()
        .ok_or_else(|| GgufError::DequantError {
            dtype: format!("{actual_dtype:?}"),
            expected: 0,
            actual: 0,
        })?;
    let db = actual_dtype.block_size().ok_or_else(|| GgufError::DequantError {
        dtype: format!("{actual_dtype:?}"),
        expected: 0,
        actual: 0,
    })?;
    let data = if use_physical {
        // 按真实 block 数截取精确字节，去掉尾部对齐 padding
        let nblocks = n / db as u64;
        let exact = nblocks * dd;
        let phys = file.tensor_data_physical(t)?;
        phys[..exact as usize].to_vec()
    } else {
        file.tensor_data(t)?
    };
    let blocks = (data.len() as u64) / dd;
    let elements = blocks * db;
    quant::dequantize(&data, actual_dtype, elements)
}

/// 将单个向量（f32）写入 `{base}_layer_{l}.bin`（逐层对照用，失败静默）。
fn dump_layer_state(base: &str, l: usize, v: &[f32]) {
    let path = format!("{base}_layer_{l}.bin");
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let _ = std::fs::write(&path, bytes);
}

/// 构建 RoPE 逆频率表：inv[i] = base^(-2i/hd)。
fn build_inv_freq(head_dim: usize, base: f64) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| base.powf(-2.0 * i as f64 / head_dim as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用恒等/全零权重构造一个最小 llama GGUF，验证 forward 可运行（输出有限）。
    #[test]
    fn test_forward_runs() {
        use crate::file::GgufFile;
        use std::io::Cursor as IoCursor;

        let p = "llama";
        let (n_layers, d, q, kv, f) = (1usize, 4usize, 2usize, 1usize, 6usize);
        let hd = d / q; // 2
        let vocab = 8u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        // 张量列表
        let tensor_names: Vec<String> = vec![
            format!("{p}.token_embd"),
            format!("{p}.0.attn.wq"),
            format!("{p}.0.attn.wk"),
            format!("{p}.0.attn.wv"),
            format!("{p}.0.attn.wo"),
            format!("{p}.0.ffn.w1"),
            format!("{p}.0.ffn.w2"),
            format!("{p}.0.ffn.w3"),
            format!("{p}.0.attn_norm"),
            format!("{p}.0.ffn_norm"),
            format!("{p}.output_norm"),
            format!("{p}.output"),
        ];
        let n_tensors = tensor_names.len() as i64;
        let n_kv: i64 = 8; // + general.alignment
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&n_kv.to_le_bytes());

        // KV 元数据
        let kvs: &[(&str, i32, &[u8])] = &[
            ("general.architecture", 8, p.as_bytes()),
            ("general.alignment", 4, &32u32.to_le_bytes()),
            ("llama.vocab_size", 4, &vocab.to_le_bytes()),
            ("llama.embedding_length", 4, &(d as u32).to_le_bytes()),
            ("llama.attention.head_count", 4, &(q as u32).to_le_bytes()),
            (
                "llama.attention.head_count_kv",
                4,
                &(kv as u32).to_le_bytes(),
            ),
            ("llama.ffn_length", 4, &(f as u32).to_le_bytes()),
            ("llama.block_count", 4, &(n_layers as u32).to_le_bytes()),
        ];
        for (key, ty, payload) in kvs {
            if *ty == 8 {
                buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(&ty.to_le_bytes());
                buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
                buf.extend_from_slice(payload);
            } else {
                buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(&ty.to_le_bytes());
                buf.extend_from_slice(payload);
            }
        }

        // 张量数据尺寸（F32，全零，列主序：shape=[rows, cols]）
        let shapes: Vec<(usize, usize)> = vec![
            (d, vocab as usize),      // token_embd: [embed_dim, vocab_size]
            (q * hd, d),              // wq: [q*hd, d]
            (kv * hd, d),             // wk: [kv*hd, d]
            (kv * hd, d),             // wv: [kv*hd, d]
            (d, q * hd),              // wo: [d, q*hd]
            (d, f),                   // w_ffn_up: [d, ffn_dim]
            (d, f),                   // w_ffn_gate: [d, ffn_dim]
            (f, d),                   // w_ffn_down: [ffn_dim, d]
            (d, 1),                   // attn_norm
            (d, 1),                   // ffn_norm
            (d, 1),                   // output_norm
            (d, vocab as usize),      // output: [d, vocab_size]
        ];
        let mut offset = 0u64;
        for (name, shape) in tensor_names.iter().zip(shapes.iter()) {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
            buf.extend_from_slice(&(shape.0 as i64).to_le_bytes());
            buf.extend_from_slice(&(shape.1 as i64).to_le_bytes());
            buf.extend_from_slice(&0i32.to_le_bytes()); // F32
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += (shape.0 * shape.1 * 4) as u64;
        }

        // 对齐填充到 32 字节边界
        let pad = (32 - (buf.len() as u64 % 32)) % 32;
        buf.extend(std::iter::repeat_n(0u8, pad as usize));

        // 数据体：全零
        let total_data = offset;
        let mut data = vec![0u8; total_data as usize];
        // 让 token_embd 行 1 全 1，使 hidden 非零（便于检测有限性）
        let embd_row1 = d;
        for i in 0..d {
            data[embd_row1 * 4 + i * 4..embd_row1 * 4 + i * 4 + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());
        }
        buf.append(&mut data);

        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let mut model = LlamaModel::new(&f).unwrap();
        let logits = model.forward(&[1u32, 0], &[0i64, 1]).unwrap();
        assert_eq!(logits.len(), vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// Prefix 分派。
    #[test]
    fn test_prefix() {
        assert_eq!(Prefix::from_arch("llama").unwrap().as_str(), "llama");
        assert_eq!(Prefix::from_arch("qwen2").unwrap().as_str(), "qwen2");
        assert!(Prefix::from_arch("bert").is_err());
    }

    /// 逆频率表首项为 1（base^0）。
    #[test]
    fn test_inv_freq() {
        let inv = build_inv_freq(8, 10000.0);
        assert_eq!(inv.len(), 4);
        assert!((inv[0] - 1.0f32).abs() < 1e-6);
    }
}
