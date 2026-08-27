//! 基础算子：GEMM / RMSNorm / Softmax / SiLU / RoPE / LayerNorm。
//!
//! 所有算子操作 f32 向量/矩阵。量化权重通过 [`super::quant::quant_matvec`]
//! 或 [`super::quant::dequantize`] 先转为 f32 再调用。

use crate::error::{GgufError, GgufResult};

// ---------- 矩阵运算 ----------

/// 行主序矩阵向量乘：`y[i] = sum_j A[i*cols+j] * x[j]`。
///
/// A 长度 = rows * cols（f32），x 长度 = cols，y 长度 = rows。
/// 可选偏置 b（长度 = rows），y[i] += b[i]。
pub fn matvec(
    a: &[f32],
    rows: u64,
    cols: u64,
    x: &[f32],
    y: &mut [f32],
    b: Option<&[f32]>,
) -> GgufResult<()> {
    let rows = rows as usize;
    let cols = cols as usize;
    if a.len() < rows * cols {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: rows as u64 * cols as u64,
            actual: a.len() as u64,
        });
    }
    if x.len() != cols {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: cols as u64,
            actual: x.len() as u64,
        });
    }
    if y.len() != rows {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: rows as u64,
            actual: y.len() as u64,
        });
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut acc = 0f32;
            let base = i * cols;
            for j in 0..cols {
                acc += a[base + j] * x[j];
            }
            *yi = acc + b.map(|b| b[i]).unwrap_or(0.0);
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            let base = i * cols;
            for j in 0..cols {
                acc += a[base + j] * x[j];
            }
            *yi = acc + b.map(|b| b[i]).unwrap_or(0.0);
        }
    }
    Ok(())
}

/// 行主序矩阵矩阵乘：`C[i,j] = sum_k A[i,k] * B[k,j]`。
///
/// A: rows_a × cols_a, B: cols_a × cols_b, C: rows_a × cols_b。
pub fn matmul(
    a: &[f32],
    rows_a: u64,
    cols_a: u64,
    b: &[f32],
    cols_b: u64,
    c: &mut [f32],
) -> GgufResult<()> {
    let (ra, ca, cb) = (rows_a as usize, cols_a as usize, cols_b as usize);
    if a.len() < ra * ca {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: (ra * ca) as u64,
            actual: a.len() as u64,
        });
    }
    if b.len() < ca * cb {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: (ca * cb) as u64,
            actual: b.len() as u64,
        });
    }
    if c.len() < ra * cb {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: (ra * cb) as u64,
            actual: c.len() as u64,
        });
    }
    for i in 0..ra {
        for j in 0..cb {
            let mut acc = 0f32;
            for k in 0..ca {
                acc += a[i * ca + k] * b[k * cb + j];
            }
            c[i * cb + j] = acc;
        }
    }
    Ok(())
}

/// 列主序矩阵向量乘：`y[i] = sum_j A[j*rows+i] * x[j]`。
///
/// A 长度 = rows * cols（f32），内存布局为列主序（dim[0] 连续）。
/// 逻辑矩阵 M[i,j] = A[j*rows + i]（rows × cols）。
/// x 长度 = cols，y 长度 = rows。
pub fn matvec_colmajor(
    a: &[f32],
    rows: u64,
    cols: u64,
    x: &[f32],
    y: &mut [f32],
    b: Option<&[f32]>,
) -> GgufResult<()> {
    let rows = rows as usize;
    let cols = cols as usize;
    if a.len() < rows * cols {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: rows as u64 * cols as u64,
            actual: a.len() as u64,
        });
    }
    if x.len() != cols {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: cols as u64,
            actual: x.len() as u64,
        });
    }
    if y.len() != rows {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: rows as u64,
            actual: y.len() as u64,
        });
    }
    // 列主序：a[j*rows+i] 对固定 j 连续（stride=1），顺序扫描即缓存友好。
    // 注意：不可用 rayon 对 i 并行（a[j*rows+i] 跨步访问 → cache miss 灾难）。
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..cols {
            acc += a[j * rows + i] * x[j];
        }
        *yi = acc + b.map(|b| b[i]).unwrap_or(0.0);
    }
    Ok(())
}

/// 列主序矩阵转置向量乘：`y[i] = sum_j A[j + i*dim0] * x[j]`。
///
/// GGUF 张量 shape=[dim0, dim1]，列主序存储（dim0 连续）。
/// 逻辑矩阵 A 为 [dim0, dim1]（dim0 行，dim1 列）。
/// 此函数计算 `y = A^T * x`，其中 x 长度 = dim0，y 长度 = dim1。
/// `y[i] = sum_j A[j, i] * x[j] = sum_j a[j + i*dim0] * x[j]`。
///
/// 用于 FFN up/gate（shape=[d, f]，计算 [f] 输出）和 FFN down（shape=[f, d]，计算 [d] 输出）。
pub fn matvec_colmajor_trans(
    a: &[f32],
    dim0: u64,
    dim1: u64,
    x: &[f32],
    y: &mut [f32],
    b: Option<&[f32]>,
) -> GgufResult<()> {
    let dim0 = dim0 as usize;
    let dim1 = dim1 as usize;
    if a.len() < dim0 * dim1 {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: dim0 as u64 * dim1 as u64,
            actual: a.len() as u64,
        });
    }
    if x.len() != dim0 {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: dim0 as u64,
            actual: x.len() as u64,
        });
    }
    if y.len() != dim1 {
        return Err(GgufError::DequantError {
            dtype: "F32".into(),
            expected: dim1 as u64,
            actual: y.len() as u64,
        });
    }
    // a[j + i*dim0] 对固定 i 连续（stride=1），rayon 并行化 i 无 cache miss。
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut acc = 0f32;
            let base = i * dim0;
            for j in 0..dim0 {
                acc += a[base + j] * x[j];
            }
            *yi = acc + b.map(|b| b[i]).unwrap_or(0.0);
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            for j in 0..dim0 {
                acc += a[j + i * dim0] * x[j];
            }
            *yi = acc + b.map(|b| b[i]).unwrap_or(0.0);
        }
    }
    Ok(())
}

// ---------- 激活函数 ----------

/// SiLU (SwiGLU)：x * sigmoid(x)。
///
/// 用分支实现 sigmoid 以避免 exp 上溢：
/// - x >= 0: sigmoid = 1 / (1 + exp(-x))，exp(-x) ∈ (0,1]
/// - x <  0: sigmoid = exp(x) / (1 + exp(x))，exp(x) ∈ (0,1)
/// 两种情形指数参数都不超过 0，永不产生 +∞。
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let sig = if *v >= 0.0 {
            1.0 / (1.0 + (-*v).exp())
        } else {
            let e = v.exp();
            e / (1.0 + e)
        };
        *v = *v * sig;
    }
}

/// GELU (tanh approximation)。
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = 0.797_884_6 * (*v + 0.044715 * (*v).powi(3));
        *v = 0.5 * *v * (1.0 + t.tanh());
    }
}

// ---------- 归一化 ----------

/// RMSNorm：out = x / sqrt(mean(x^2) + eps) * weight。
///
/// x 和 weight 长度相同（每元素 1 个 weight）。
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    if x.len() != weight.len() {
        return;
    }
    // 用 f64 累加平方和，避免 f32 在长序列/深堆叠下的精度漂移
    // （llama.cpp / candle 均用 f64 sum-of-squares）。
    let n = x.len() as f64;
    let ss: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let scale = (ss / n + f64::from(eps)).recip().sqrt() as f32;
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * scale * weight[i];
    }
}

/// LayerNorm（pre-norm）：out = (x - mean) / sqrt(var + eps) * weight + bias。
///
/// x, weight, bias 长度相同。
pub fn layernorm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    if x.is_empty() || x.len() != weight.len() || x.len() != bias.len() {
        return;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = (var + eps).recip().sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv_std * weight[i] + bias[i];
    }
}

// ---------- Softmax ----------

/// 稳定 Softmax：out[i] = exp(x[i] - max) / sum(exp(x - max))。
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    // 注意：f32::MIN 是最小正数（1.17e-38），不是最小值。
    // 全负数的 attention scores 若用 f32::MIN 初始化 max，会导致 max≈0 而非真实负值，
    // exp(scores - max) 计算错误，24 层累积后输出乱码。须用 NEG_INFINITY。
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v = (*v - max).exp() * inv_sum;
    }
}

// ---------- RoPE ----------

/// 旋转位置编码（GPT-NeoX / LLaMA 风格，rotate_half）。
///
/// x 长度 = n * head_dim（n 个 token，每个 head_dim 维）。
/// freqs 为逆频率表：freqs[i] = 1 / base^(2i/head_dim)，长度 = head_dim/2。
/// 对每个 token 的每对维度 (x[i], x[i + head_dim/2]) 旋转 pos * freqs[i] 弧度
/// （Qwen2/LLaMA 的 rotate_half 约定，配对间隔 head_dim/2，而非相邻维）。
pub fn rope(x: &mut [f32], positions: &[i64], freqs: &[f32]) {
    if x.is_empty() || positions.is_empty() || freqs.is_empty() {
        return;
    }
    let half = freqs.len();
    let head_dim = half * 2;
    let n = x.len() / head_dim;
    for tok in 0..n {
        let pos = positions.get(tok).copied().unwrap_or(0) as f32;
        let base = tok * head_dim;
        for i in 0..half {
            let angle = pos * freqs[i];
            let (sin, cos) = angle.sin_cos();
            let a = x[base + i];
            let b = x[base + i + half];
            x[base + i] = a * cos - b * sin;
            x[base + i + half] = b * cos + a * sin;
        }
    }
}

/// 构建 RoPE 逆频率表：freqs[i] = 1 / base^(2i/dim)。
pub fn rope_freqs(dim: usize, base: f32, max_seq_len: usize) -> Vec<Vec<f32>> {
    let dim_f = dim as f32;
    let inv_freq: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / base.powf((2.0 * i as f32) / dim_f))
        .collect();
    let mut freqs = Vec::with_capacity(max_seq_len);
    for pos in 0..max_seq_len {
        let p = pos as f32;
        freqs.push(inv_freq.iter().map(|&f| p * f).collect());
    }
    freqs
}

// ---------- 线性层（含偏置）----------

/// 线性层：y = A * x + b（行主序）。
///
/// A: out_dim × in_dim, x: in_dim, b: out_dim (可选)。
pub fn linear(
    a: &[f32],
    out_dim: u64,
    in_dim: u64,
    x: &[f32],
    b: Option<&[f32]>,
) -> GgufResult<Vec<f32>> {
    let mut y = vec![0f32; out_dim as usize];
    matvec(a, out_dim, in_dim, x, &mut y, b)?;
    Ok(y)
}

// ---------- 测试 ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_2x3() {
        // A = [[1,2,3],[4,5,6]] (2x3), x = [1,1,1], b = [10,20]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let b = vec![10.0, 20.0];
        let mut y = vec![0.0; 2];
        matvec(&a, 2, 3, &x, &mut y, Some(&b)).unwrap();
        assert!((y[0] - (1.0 + 2.0 + 3.0 + 10.0)).abs() < 1e-6);
        assert!((y[1] - (4.0 + 5.0 + 6.0 + 20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_no_bias() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let x = vec![3.0, 7.0];
        let mut y = vec![0.0; 2];
        matvec(&a, 2, 2, &x, &mut y, None).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_dim_mismatch() {
        let a = vec![1.0; 4];
        let x = vec![1.0; 3];
        let mut y = vec![0.0; 2];
        assert!(matvec(&a, 2, 3, &x, &mut y, None).is_err());
    }

    #[test]
    fn test_matmul_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A*B = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        matmul(&a, 2, 2, &b, 2, &mut c).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0f32, 1.0, -1.0, 10.0];
        silu(&mut x);
        // silu(0) = 0, silu(1) ≈ 0.731059, silu(-1) ≈ -0.268941
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 1.0f32 / (1.0f32 + (-1.0f32).exp())).abs() < 1e-5);
        assert!((x[2] - (-1.0f32) / (1.0f32 + 1.0f32.exp())).abs() < 1e-5);
        assert!((x[3] - 10.0f32 / (1.0f32 + (-10.0f32).exp())).abs() < 1e-5);
    }

    /// SiLU 大值溢出防护：极大正/负值不得产生 NaN 或 Inf。
    #[test]
    fn test_silu_no_overflow() {
        let mut x = vec![100.0f32, -100.0, 1000.0, -1000.0, -1e10];
        silu(&mut x);
        for (i, v) in x.iter().enumerate() {
            assert!(v.is_finite(), "silu overflow at {i}: {v}");
        }
        // silu(-100) ≈ -100 * exp(-100) ≈ 0（负无穷方向趋于 0）
        assert!(x[1].abs() < 1e-30, "silu(-100) should be ~0, got {}", x[1]);
        // silu(100) ≈ 100（正方向饱和）
        assert!((x[0] - 100.0).abs() < 1e-3, "silu(100) should be ~100, got {}", x[0]);
    }

    #[test]
    fn test_gelu() {
        let mut x = vec![0.0f32, 1.0];
        gelu(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 0.8413).abs() < 1e-3); // gelu(1) ≈ 0.8413
    }

    #[test]
    fn test_rmsnorm() {
        // x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-6
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        rmsnorm(&mut x, &w, 1e-6);
        // mean(x^2) = (1+4+9+16)/4 = 7.5, scale = 1/sqrt(7.5) ≈ 0.3651
        // x_norm = [0.3651, 0.7303, 1.0954, 1.4606]
        let scale = (7.5f32 + 1e-6f32).recip().sqrt();
        assert!((x[0] - 1.0f32 * scale).abs() < 1e-6);
        assert!((x[3] - 4.0f32 * scale).abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm_with_weight() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![2.0f32, 1.0, 1.0, 1.0];
        rmsnorm(&mut x, &w, 1e-6);
        let scale = (7.5f32 + 1e-6f32).recip().sqrt();
        assert!((x[0] - 1.0f32 * scale * 2.0f32).abs() < 1e-6);
        assert!((x[1] - 2.0f32 * scale).abs() < 1e-6);
    }

    #[test]
    fn test_layernorm() {
        // x = [1, 2, 3, 4], weight = [1,1,1,1], bias = [0,0,0,0]
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let b = vec![0.0f32; 4];
        layernorm(&mut x, &w, &b, 1e-6);
        // mean = 2.5, var = 1.25, std ≈ 1.118
        // x_norm = [(1-2.5)/1.118, (2-2.5)/1.118, ...] ≈ [-1.342, -0.447, 0.447, 1.342]
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv = (var + 1e-6f32).recip().sqrt();
        assert!((x[0] - (1.0f32 - mean) * inv).abs() < 1e-5);
        assert!((x[3] - (4.0f32 - mean) * inv).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax(&mut x);
        // sum should be 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // max prob at index 2
        assert!(x[2] > x[1] && x[1] > x[0]);
        // known values: exp(3-max)/sum ≈ 0.6652
        let e1 = (1.0f32 - 3.0f32).exp();
        let e2 = (2.0f32 - 3.0f32).exp();
        let e3 = (3.0f32 - 3.0f32).exp();
        let s = e1 + e2 + e3;
        assert!((x[0] - e1 / s).abs() < 1e-6);
        assert!((x[2] - e3 / s).abs() < 1e-6);
    }

    #[test]
    fn test_rope_basic() {
        // 2 tokens, head_dim=4 (2 freq pairs)
        // token 0 at pos 0: no rotation
        // token 1 at pos 1: rotate by freqs
        let inv_freq = vec![1.0f32, 0.001]; // 2 freqs → head_dim=4（half-split 布局）
        let mut x = vec![
            // token 0 (pos 0)：[x0, x2, x1, x3]
            1.0, 0.0, 0.0, 1.0,
            // token 1 (pos 1)：[x0, x2, x1, x3]
            1.0, 0.0, 0.0, 1.0,
        ];
        let positions = vec![0i64, 1];
        rope(&mut x, &positions, &inv_freq);
        // token 0 unchanged (pos 0 → angle 0 → cos=1, sin=0)
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
        assert!((x[2] - 0.0).abs() < 1e-6);
        assert!((x[3] - 1.0).abs() < 1e-6);
        // token 1（half-split）：
        //   pair 0: a=x[4]=1.0, b=x[6]=0.0, angle=1*1.0
        //     x[4]=a*cos0-b*sin0, x[6]=b*cos0+a*sin0
        //   pair 1: a=x[5]=0.0, b=x[7]=1.0, angle=1*0.001
        //     x[5]=a*cos1-b*sin1, x[7]=b*cos1+a*sin1
        let (sin0, cos0) = 1.0f32.sin_cos();
        let (sin1, cos1) = 0.001f32.sin_cos();
        assert!((x[4] - (1.0 * cos0 - 0.0 * sin0)).abs() < 1e-6);
        assert!((x[5] - (0.0 * cos1 - 1.0 * sin1)).abs() < 1e-6);
        assert!((x[6] - (0.0 * cos0 + 1.0 * sin0)).abs() < 1e-6);
        assert!((x[7] - (1.0 * cos1 + 0.0 * sin1)).abs() < 1e-6);
    }

    #[test]
    fn test_rope_freqs() {
        let freqs = rope_freqs(8, 10000.0, 5);
        assert_eq!(freqs.len(), 5);
        assert_eq!(freqs[0].len(), 4);
        // pos 0: all zeros
        assert!(freqs[0].iter().all(|&f| f.abs() < 1e-6));
        // pos 1: freqs[i] = 1 * inv_freq[i]
        // inv_freq[0] = 1/10000^(0/8) = 1.0
        assert!((freqs[1][0] - 1.0).abs() < 1e-6);
        // inv_freq[1] = 1/10000^(2/8) = 10000^(-0.25) ≈ 0.31622777
        assert!((freqs[1][1] - 1.0f32 / 10000.0f32.powf(0.25f32)).abs() < 1e-6);
    }

    #[test]
    fn test_linear() {
        // A = [[1,2],[3,4]], x = [1,1], b = [10,20]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let b = vec![10.0, 20.0];
        let y = linear(&a, 2, 2, &x, Some(&b)).unwrap();
        assert!((y[0] - (1.0 + 2.0 + 10.0)).abs() < 1e-6);
        assert!((y[1] - (3.0 + 4.0 + 20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_3x2_times_2x3() {
        // A (3x2) = [[1,2],[3,4],[5,6]], B (2x3) = [[7,8,9],[10,11,12]]
        // C = A*B (3x3):
        // C[0] = [1*7+2*10, 1*8+2*11, 1*9+2*12] = [27, 30, 33]
        // C[1] = [3*7+4*10, 3*8+4*11, 3*9+4*12] = [61, 68, 75]
        // C[2] = [5*7+6*10, 5*8+6*11, 5*9+6*12] = [95, 106, 117]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 9];
        matmul(&a, 3, 2, &b, 3, &mut c).unwrap();
        let expected = [27.0, 30.0, 33.0, 61.0, 68.0, 75.0, 95.0, 106.0, 117.0];
        for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "c[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_softmax_stability() {
        // Large values that would overflow exp
        let mut x = vec![1000.0f32, 1001.0, 1002.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
