//! 量化反量化与量化原生 GEMM。
//!
//! 支持 F32 / F16 / BF16 及 GGML 全部量化类型（Q4_0~Q8_0、Q2_K~Q8_K）。
//! 布局与公式对照 llama.cpp `ggml-quants.c`（master）。

use crate::error::{GgufError, GgufResult};
use crate::types::GgmlType;

// ---------- f16 / bf16 编解码 ----------

/// f16 解码（IEEE 754 半精度 → f32）。
pub fn f16_to_f32(b: &[u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(*b) as u32;
    let sign = (bits & 0x8000) != 0;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let v: f32 = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32 / 1024.0) * 2f32.powi(-14)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            return f32::NAN;
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign {
        -v
    } else {
        v
    }
}

/// bf16 解码（bfloat16 → f32）。
pub fn bf16_to_f32(b: &[u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)
}

/// f16 编码（f32 → IEEE 754 半精度，round-to-nearest）。
pub fn f32_to_f16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits & 0x8000_0000) >> 16) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    let out: u16 = if exp == 0xFF {
        // inf / NaN
        if mant != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        }
    } else if exp > 0x8E {
        // 上溢 → inf
        sign | 0x7C00
    } else if exp > 0x70 {
        // 正规数（f32 exp > 0x70 才映射到 f16 正规数）
        // e = exp - 0x7F + 15 = exp - 0x70
        let e = (exp - 0x70) as u16;
        let m10 = (mant >> 13) as u16;
        sign | (e << 10) | m10
    } else if exp == 0 {
        // f32 次正规值 = M*2^-149, 总是下溢到 0
        sign
    } else {
        // exp in [1, 0x70]: 正规数，映射到 f16 正规数
        // m16 = round((M | 0x800000) >> (126 - exp))
        let raw = mant | 0x800000u32;
        let shift = (126 - exp) as usize; // 14..=125
        let val = raw >> shift;
        let rounding_bit = (raw >> (shift - 1)) & 1;
        let m16 = (val + rounding_bit) as u16;
        if m16 == 0 {
            sign
        } else {
            m16 | sign
        }
    };
    out.to_le_bytes()
}

/// 从字节切片读取第 `byte_off` 字节处的 f16 值。
fn f16_at(data: &[u8], byte_off: usize) -> f32 {
    f16_to_f32(&data[byte_off..byte_off + 2].try_into().unwrap())
}

/// 从字节切片读取第 `byte_off` 字节处的 bf16 值。
fn bf16_at(data: &[u8], byte_off: usize) -> f32 {
    bf16_to_f32(&data[byte_off..byte_off + 2].try_into().unwrap())
}

// ---------- 位操作辅助 ----------

/// 取 4-bit scale 数组（16 字节 permute 存储）中第 j 个 scale（0..15）。
/// j=0 → s[15] 低 4 位, j=1 → s[15] 高 4 位, j=2 → s[14] 低 4 位, ...
#[inline]
fn k_scale4(s: &[u8], j: usize) -> u32 {
    let byte = 15 - j / 2;
    if j.is_multiple_of(2) {
        (s[byte] & 0xF) as u32
    } else {
        ((s[byte] >> 4) & 0xF) as u32
    }
}

/// Q4_K/Q5_K: 从 12 字节 scales 数组中解码第 j 组（0..16）的 6-bit scale 和 6-bit min。
/// 参考 llama.cpp `get_scale_min_k4`（ggml-quants.c）：
///   j < 4:  scale = q[j] & 0x3F,        min = q[j+4] & 0x3F
///   j >= 4: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///           min   = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
fn k_scale_min_6bit(scales: &[u8], j: usize) -> (u32, u32) {
    if j < 4 {
        ((scales[j] & 0x3F) as u32, (scales[j + 4] & 0x3F) as u32)
    } else {
        let sc = ((scales[j + 4] & 0x0F) as u32) | (((scales[j - 4] >> 6) as u32) << 4);
        let mn = ((scales[j + 4] >> 4) as u32) | (((scales[j] >> 6) as u32) << 4);
        (sc, mn)
    }
}

// ---------- 基础量化 block 反量化（32 元素）----------

fn deq_q4_0(block: &[u8], out: &mut [f32]) {
    let d = f16_at(block, 0);
    for i in 0..16 {
        let byte = block[2 + i];
        out[2 * i] = d * (((byte & 0xF) as i8) - 8) as f32;
        out[2 * i + 1] = d * (((byte >> 4) as i8) - 8) as f32;
    }
}

fn deq_q4_1(block: &[u8], out: &mut [f32]) {
    let d = f16_at(block, 0);
    let m = f16_at(block, 2);
    for i in 0..16 {
        let byte = block[4 + i];
        out[2 * i] = d * (byte & 0xF) as f32 + m;
        out[2 * i + 1] = d * (byte >> 4) as f32 + m;
    }
}

fn deq_q5_0(block: &[u8], out: &mut [f32]) {
    // Q5_0 layout: d(2) + qh(4) + qs(16) = 22 bytes, 32 elements
    // 对照 llama.cpp ggml-quants.c 官方展开：
    //   x0 = (qs[j] & 0x0F) | xh_0  - 16  (j+0..qk/2-1)
    //   x1 = (qs[j] >> 4)   | xh_1  - 16  (j+qk/2..qk-1)
    //   xh_0 = ((qh >> j) << 4) & 0x10    (j+0..j+15)
    //   xh_1 = ((qh >> (j+12)) & 0x10)    (j+12..j+27)
    let d = f16_at(block, 0);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for i in 0..16 {
        let byte = block[6 + i];
        let xh_0 = ((qh >> i) << 4) & 0x10; // bits 0..15
        let xh_1 = (qh >> (i + 12)) & 0x10; // bits 12..27
        let lo = (((byte & 0xF) as u32) | xh_0) as i32 - 16;
        let hi = (((byte >> 4) as u32) | xh_1) as i32 - 16;
        out[i] = d * lo as f32;
        out[16 + i] = d * hi as f32;
    }
}

fn deq_q5_1(block: &[u8], out: &mut [f32]) {
    // Q5_1 layout: d(2) + m(2) + qh(4) + qs(16) = 24 bytes, 32 elements
    // 官方展开顺序同 Q5_0
    let d = f16_at(block, 0);
    let m = f16_at(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for i in 0..16 {
        let byte = block[8 + i];
        let xh_0 = ((qh >> i) << 4) & 0x10;
        let xh_1 = (qh >> (i + 12)) & 0x10;
        let lo = ((byte & 0xF) as u32) | xh_0;
        let hi = ((byte >> 4) as u32) | xh_1;
        out[i] = d * lo as f32 + m;
        out[16 + i] = d * hi as f32 + m;
    }
}

fn deq_q8_0(block: &[u8], out: &mut [f32]) {
    let d = f16_at(block, 0);
    for i in 0..32 {
        out[i] = d * (block[2 + i] as i8) as f32;
    }
}

// ---------- K-quant super-block 反量化（256 元素）----------

fn deq_q2_k(block: &[u8], out: &mut [f32]) {
    // Q2_K: 84 bytes = scales[16]@0 + qs[64]@16 + d@80 + dmin@82
    let d = f16_at(block, 80);
    let dmin = f16_at(block, 82);
    for j in 0..16 {
        let s = k_scale4(&block[0..16], j) as f32;
        let m = (s - 8.0) * dmin;
        let s0 = if s < 8.0 { s * d } else { s * dmin + m };
        for i in 0..16 {
            let k = 16 * j + i;
            let q = ((block[16 + k / 4] >> ((k % 4) * 2)) & 0x3) as f32;
            out[k] = s0 * q + (if s < 8.0 { 0.0 } else { m });
        }
    }
}

/// Q3_K 反量化通用实现。
///
/// 布局对照 llama.cpp `block_q3_K`（master），参考 gguf-py `dequantize_blocks_Q3_K`：
/// - Q3_K_S: 110 bytes = hmask[32]@0 + qs[64]@32 + scales[12]@96 + d(f16)@108
/// - Q3_K_M: 114 bytes = hmask[32]@0 + qs[64]@32 + scales[12]@96 + qs_masks[4]@108 + d(f16)@112
///
/// 解码：
/// - scales[12]: 前 8 字节 = 16 个 4-bit lo_scales，后 4 字节 = 16 个 2-bit hi_scales
///   scale_i = (lo_i | (hi_i << 4)) - 32  ∈ [-32, -1]
/// - 每个元素 i (0..256):
///   ql = (qs[i/2] >> ((i%2)*2)) & 3              (2-bit)
///   qh = (hmask[i/8] >> (i%8)) & 1               (1-bit)
///   q  = (qh ^ 1) * 4 - ql                        (0..7)
///   out[i] = d * (scale_{i/16} - 32) * q
fn deq_q3_k(block: &[u8], d_off: usize, out: &mut [f32]) {
    let d = f16_at(block, d_off);
    // 解码 16 个 5-bit scales（-32..-1）
    let mut sc = [0i32; 16];
    for i in 0..16 {
        let lo = (block[96 + i / 2] >> ((i % 2) * 4)) & 0xF;
        let hi = (block[96 + 8 + i / 4] >> ((i % 4) * 2)) & 0x3;
        sc[i] = (lo | (hi << 4)) as i32 - 32;
    }
    for i in 0..256 {
        // ql: 2-bit，每字节 4 个（qs 共 64 字节，offset 32）
        let ql = ((block[32 + i / 4] >> ((i % 4) * 2)) & 3) as i32;
        // qh: 1-bit，每字节 8 个（hmask 共 32 字节，offset 0）
        let qh = ((block[i / 8] >> (i % 8)) & 1) as i32;
        let q = (qh ^ 1) * 4 - ql;
        out[i] = d * (sc[i / 16] as f32) * (q as f32);
    }
}

/// Q3_K_S 反量化（110 bytes，d @ 108）。
fn deq_q3_k_s(block: &[u8], out: &mut [f32]) {
    deq_q3_k(block, 108, out);
}

/// Q3_K_M 反量化（114 bytes，d @ 112，中间有 4 字节 qs_masks）。
fn deq_q3_k_m(block: &[u8], out: &mut [f32]) {
    deq_q3_k(block, 112, out);
}

/// Q3_K_L 反量化：布局与 Q3_K_S 相同（110 字节）。
fn deq_q3_k_l(block: &[u8], out: &mut [f32]) {
    deq_q3_k(block, 108, out);
}

fn deq_q4_k(block: &[u8], out: &mut [f32]) {
    // Q4_K: 144 bytes = dm(d@0,dmin@2) + scales[12]@4 + qs[128]@16
    // scales: 8 个 6-bit scale + 8 个 6-bit min 交织编码在 12 字节中
    // 正确布局（kekzl/imp PR#255 CPU oracle）：
    //   4 组 × 64 元素。对元素 e ∈ [0,256):
    //     group = e >> 6 (0-3), in_grp = e & 63, is_high = in_grp >> 5 (0/1)
    //     byte_in_qs = group*32 + (in_grp & 31)
    //     sub_block = group*2 + is_high
    //     nibble = qs[byte_in_qs] >> 4 if is_high else qs[byte_in_qs] & 0x0F
    let d = f16_at(block, 0);
    let dmin = f16_at(block, 2);
    for e in 0..256 {
        let group = e >> 6;
        let in_grp = e & 63;
        let is_high = in_grp >> 5;
        let byte_in_qs = group * 32 + (in_grp & 31);
        let sub_block = group * 2 + is_high;
        let (sc, mn) = k_scale_min_6bit(&block[4..16], sub_block);
        let s1 = sc as f32 * d;
        let s2 = mn as f32 * dmin;
        let byte = block[16 + byte_in_qs];
        let val = if is_high != 0 { byte >> 4 } else { byte & 0x0F };
        out[e] = s1 * val as f32 - s2;
    }
}

fn deq_q5_k(block: &[u8], out: &mut [f32]) {
    // Q5_K: 176 bytes = dm(d@0,dmin@2) + scales[12]@4 + qh[32]@16 + qs[128]@48
    // scales: 8 个 6-bit scale + 8 个 6-bit min 交织编码在 12 字节中
    // 8 个子块 × 32 元素 = 256
    let d = f16_at(block, 0);
    let dmin = f16_at(block, 2);
    for j in 0..8 {
        let (sc, mn) = k_scale_min_6bit(&block[4..16], j);
        let s1 = sc as f32 * d;
        let s2 = mn as f32 * dmin;
        for i in 0..32 {
            let k = 32 * j + i;
            let byte = block[48 + k / 2];
            let byte_idx = k >> 1;
            let qh_bit = ((block[16 + byte_idx / 8] >> (byte_idx % 8)) & 1) as u32;
            let q = if k & 1 == 0 {
                ((byte & 0xF) as u32) | qh_bit << 4
            } else {
                ((byte >> 4) as u32) | qh_bit << 4
            };
            out[k] = s1 * q as f32 - s2;
        }
    }
}

fn deq_q6_k(block: &[u8], out: &mut [f32]) {
    // Q6_K: 210 bytes = ql[128]@0 + qh[64]@128 + scales[16 int8]@192 + d(f16)@208
    // 参考 ggml dequantize_row_q6_k：256 元素分 2 个 128 半区 (n=0,128)
    //   每半区: ql 用 64 字节, qh 用 32 字节, sc 用 8 个 int8
    //   ql[l] 低 4-bit -> 元素 l (q1), 高 4-bit -> 元素 l+64 (q3)
    //   ql[l+32] 低 4-bit -> 元素 l+32 (q2), 高 4-bit -> 元素 l+96 (q4)
    //   qh[l] 含 4 个 2-bit (shift 0/2/4/6) 分别给 q1/q2/q3/q4
    //   sc 步长 2: sc[is]->q1, sc[is+2]->q2, sc[is+4]->q3, sc[is+6]->q4, is=l/16
    let d = f16_at(block, 208);
    for n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let ql0 = block[n * 64 + l];
            let ql1 = block[n * 64 + 32 + l];
            let qh = block[128 + n * 32 + l];
            let sc = &block[192 + n * 8 .. 192 + n * 8 + 8];
            let q1 = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            let q2 = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            let q3 = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            let q4 = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            let base = n * 128;
            // sc 是有符号 int8（-128~127），须先 as i8 再 as i32，
            // 直接 u8 as i32 会把 >127 的 scale 误读为正数（应为负数）。
            out[base + l] = d * (sc[is] as i8) as f32 * (q1 as i32 - 32) as f32;
            out[base + l + 32] = d * (sc[is + 2] as i8) as f32 * (q2 as i32 - 32) as f32;
            out[base + l + 64] = d * (sc[is + 4] as i8) as f32 * (q3 as i32 - 32) as f32;
            out[base + l + 96] = d * (sc[is + 6] as i8) as f32 * (q4 as i32 - 32) as f32;
        }
    }
}

fn deq_q8_k(block: &[u8], out: &mut [f32]) {
    // Q8_K: 292 bytes = d(f32)@0 + qs[256]@4 + bsums[16]@260
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    for i in 0..256 {
        out[i] = d * (block[4 + i] as i8) as f32;
    }
}

// ---------- 单 block 反量化入口 ----------

fn dtype_name(dt: GgmlType) -> String {
    format!("{dt:?}")
}

/// 反量化单个 block。
///
/// `out` 长度须 >= 该 dtype 的 block 元素数（32 或 256，浮点为 1）。
/// `data` 长度须 >= 该 dtype 的 block 字节数。
pub fn dequantize_block(data: &[u8], dtype: GgmlType, out: &mut [f32]) -> GgufResult<()> {
    let block_bytes = dtype.block_bytes().ok_or_else(|| GgufError::DequantError {
        dtype: dtype_name(dtype),
        expected: 0,
        actual: data.len() as u64,
    })? as usize;
    if data.len() < block_bytes {
        return Err(GgufError::DequantError {
            dtype: dtype_name(dtype),
            expected: block_bytes as u64,
            actual: data.len() as u64,
        });
    }
    match dtype {
        GgmlType::F32 => {
            out[0] = f32::from_le_bytes(data[0..4].try_into().unwrap());
        }
        GgmlType::F16 => out[0] = f16_at(data, 0),
        GgmlType::BF16 => out[0] = bf16_at(data, 0),
        GgmlType::Q4_0 => deq_q4_0(data, out),
        GgmlType::Q4_1 => deq_q4_1(data, out),
        GgmlType::Q5_0 => deq_q5_0(data, out),
        GgmlType::Q5_1 => deq_q5_1(data, out),
        GgmlType::Q8_0 => deq_q8_0(data, out),
        GgmlType::Q2_K => deq_q2_k(data, out),
        GgmlType::Q3_K_S => deq_q3_k_s(data, out),
        GgmlType::Q3_K_M => deq_q3_k_m(data, out),
        GgmlType::Q3_K_L => deq_q3_k_l(data, out),
        GgmlType::Q4_K => deq_q4_k(data, out),
        GgmlType::Q5_K => deq_q5_k(data, out),
        GgmlType::Q6_K => deq_q6_k(data, out),
        GgmlType::Q8_K => deq_q8_k(data, out),
        _ => {
            return Err(GgufError::DequantError {
                dtype: dtype_name(dtype),
                expected: 0,
                actual: 0,
            });
        }
    }
    Ok(())
}

/// 将量化张量数据反量化为 f32 向量。
///
/// `data` 长度必须 >= `tensor.data_size()`，`n` 为元素总数。
pub fn dequantize(data: &[u8], dtype: GgmlType, n: u64) -> GgufResult<Vec<f32>> {
    let bs = dtype.block_size().ok_or_else(|| GgufError::DequantError {
        dtype: dtype_name(dtype),
        expected: 0,
        actual: 0,
    })?;
    if !n.is_multiple_of(bs) {
        return Err(GgufError::InvalidTensorShape {
            name: String::new(),
            elements: n,
            block: bs,
        });
    }
    let bb = dtype.block_bytes().unwrap() as usize;
    let blocks = (n / bs) as usize;
    let need = blocks * bb;
    if data.len() < need {
        return Err(GgufError::DequantError {
            dtype: dtype_name(dtype),
            expected: need as u64,
            actual: data.len() as u64,
        });
    }
    let mut out = vec![0f32; n as usize];
    let bs_us = bs as usize;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // 取精确长度切片，par_chunks / par_chunks_mut 各产生 blocks 个 chunk，zip 后并行
        let _ = &data[..need]
            .par_chunks(bb)
            .zip(out.as_mut_slice().par_chunks_mut(bs_us))
            .try_for_each(|(src, dst)| dequantize_block(src, dtype, dst))?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..blocks {
            dequantize_block(
                &data[b * bb..(b + 1) * bb],
                dtype,
                &mut out[b * bs_us..(b + 1) * bs_us],
            )?;
        }
    }
    Ok(out)
}

// ---------- 量化原生 matvec ----------

fn row_bytes(dtype: GgmlType, cols: u64) -> GgufResult<usize> {
    let bs = dtype.block_size().ok_or_else(|| GgufError::DequantError {
        dtype: dtype_name(dtype),
        expected: 0,
        actual: 0,
    })? as usize;
    let bb = dtype.block_bytes().ok_or_else(|| GgufError::DequantError {
        dtype: dtype_name(dtype),
        expected: 0,
        actual: 0,
    })? as usize;
    if !(cols as usize).is_multiple_of(bs) {
        return Err(GgufError::InvalidTensorShape {
            name: String::new(),
            elements: cols,
            block: bs as u64,
        });
    }
    Ok((cols as usize / bs) * bb)
}

fn dot_row(w: &[u8], dtype: GgmlType, row_start: usize, cols: usize, x: &[f32]) -> f32 {
    match dtype {
        GgmlType::F32 => {
            let mut acc = 0f32;
            for c in 0..cols {
                let v = f32::from_le_bytes(
                    w[row_start + c * 4..row_start + c * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                acc += v * x[c];
            }
            acc
        }
        GgmlType::F16 => {
            let mut acc = 0f32;
            for (c, x_c) in x.iter().enumerate() {
                acc += f16_at(w, row_start + c * 2) * x_c;
            }
            acc
        }
        GgmlType::BF16 => {
            let mut acc = 0f32;
            for (c, x_c) in x.iter().enumerate() {
                acc += bf16_at(w, row_start + c * 2) * x_c;
            }
            acc
        }
        _ => {
            let bs = dtype.block_size().unwrap() as usize;
            let bb = dtype.block_bytes().unwrap() as usize;
            let mut block = vec![0f32; bs];
            let mut acc = 0f32;
            for b in 0..(cols / bs) {
                dequantize_block(
                    &w[row_start + b * bb..row_start + (b + 1) * bb],
                    dtype,
                    &mut block,
                )
                .expect("block size validated");
                for c in 0..bs {
                    acc += block[c] * x[b * bs + c];
                }
            }
            acc
        }
    }
}

/// 量化原生矩阵向量乘：`y[i] = sum_j W[i,j] * x[j]`。
///
/// W 为量化存储（行主序，rows × cols），x 为 f32 向量（长度 cols），y 长度 rows。
/// 对量化类型按 block 反量化 + 累加，不物化完整 f32 矩阵。
pub fn quant_matvec(
    w: &[u8],
    dtype: GgmlType,
    rows: u64,
    cols: u64,
    x: &[f32],
    y: &mut [f32],
) -> GgufResult<()> {
    let rb = row_bytes(dtype, cols)?;
    if x.len() != cols as usize {
        return Err(GgufError::DequantError {
            dtype: dtype_name(dtype),
            expected: cols,
            actual: x.len() as u64,
        });
    }
    if y.len() != rows as usize {
        return Err(GgufError::DequantError {
            dtype: dtype_name(dtype),
            expected: rows,
            actual: y.len() as u64,
        });
    }
    if (rows as usize) * rb > w.len() {
        return Err(GgufError::DequantError {
            dtype: dtype_name(dtype),
            expected: (rows as usize) as u64 * rb as u64,
            actual: w.len() as u64,
        });
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(r, yi)| {
            *yi = dot_row(w, dtype, r * rb, cols as usize, x);
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (r, yi) in y.iter_mut().enumerate() {
            *yi = dot_row(w, dtype, r * rb, cols as usize, x);
        }
    }
    Ok(())
}

// ---------- 测试 ----------

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立的 f16 编码（仅用于测试，与 f32_to_f16 无关）。
    fn f16b(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let sign = ((bits & 0x8000_0000) >> 16) as u16;
        let exp = (bits >> 23) & 0xFF;
        let mant = bits & 0x7FFFFF;
        let out: u16 = if exp == 0xFF {
            if mant != 0 {
                sign | 0x7E00
            } else {
                sign | 0x7C00
            }
        } else if exp > 0x8E {
            sign | 0x7C00
        } else if exp > 0x70 {
            // e = exp - 0x7F + 15 = exp - 0x70
            let e = (exp - 0x70) as u16;
            let m10 = (mant >> 13) as u16;
            sign | (e << 10) | m10
        } else if exp == 0 {
            sign
        } else {
            let raw = mant | 0x800000u32;
            let shift = (126 - exp) as usize;
            let val = raw >> shift;
            let rounding_bit = (raw >> (shift - 1)) & 1;
            let m16 = (val + rounding_bit) as u16;
            if m16 == 0 {
                sign
            } else {
                m16 | sign
            }
        };
        out.to_le_bytes()
    }

    /// 测试辅助：f16b 编码后解码回 f32，模拟 deq 中的 f16 精度。
    fn f16_at_test(v: f32) -> f32 {
        let b = f16b(v);
        f16_at(&b, 0)
    }

    /// f16 往返精度（可精确表示的值）。
    #[test]
    fn test_f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.25, -100.0, 65504.0, 6.1035156e-5] {
            let dec = f16_to_f32(&f16b(v));
            assert!((dec - v).abs() < 1e-5 * v.abs().max(1.0), "v={v} dec={dec}");
        }
    }

    /// f16 边界：inf / NaN / 负零 / 上溢。
    #[test]
    fn test_f16_boundaries() {
        assert_eq!(f16_to_f32(&f16b(f32::INFINITY)), f32::INFINITY);
        assert!(f16_to_f32(&f16b(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(&f16b(-0.0)).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(&f16b(1e30)), f32::INFINITY);
    }

    /// bf16 往返（截断 mantissa 16 位）。
    #[test]
    fn test_bf16_roundtrip() {
        for v in [0.0f32, 1.0, -2.5, 123.0, -0.25, 1.5e5] {
            let bits = v.to_bits() & 0xFFFF0000;
            let enc: [u8; 2] = ((bits >> 16) as u16).to_le_bytes();
            assert_eq!(bf16_to_f32(&enc), f32::from_bits(bits));
        }
    }

    /// Q4_0 反量化与手算对比。
    #[test]
    fn test_q4_0() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16b(0.25));
        for (i, b) in block[2..].iter_mut().enumerate() {
            let v = i as u8;
            *b = v | (v << 4);
        }
        let mut out = [0f32; 32];
        dequantize_block(&block, GgmlType::Q4_0, &mut out).unwrap();
        for i in 0..16 {
            let q = i as i8;
            assert!((out[2 * i] - 0.25 * (q - 8) as f32).abs() < 1e-6);
            assert!((out[2 * i + 1] - 0.25 * (q - 8) as f32).abs() < 1e-6);
        }
    }

    /// Q4_1 反量化。
    #[test]
    fn test_q4_1() {
        let mut block = [0u8; 20];
        block[0..2].copy_from_slice(&f16b(0.5)); // d=0.5
        block[2..4].copy_from_slice(&f16b(0.0)); // m=0
        for (i, b) in block[4..].iter_mut().enumerate() {
            let lo = i as u8;
            let hi = (i as u8).wrapping_add(1) & 0xF; // 保持 4-bit 范围内
            *b = lo | (hi << 4);
        }
        let mut out = [0f32; 32];
        dequantize_block(&block, GgmlType::Q4_1, &mut out).unwrap();
        for i in 0..16 {
            let lo = i as u32;
            let hi = (i + 1) & 0xF;
            assert!(
                (out[2 * i] - 0.5 * lo as f32).abs() < 1e-6,
                "i={i} lo={lo} out={}",
                out[2 * i]
            );
            assert!(
                (out[2 * i + 1] - 0.5 * hi as f32).abs() < 1e-6,
                "i={i} hi={hi} out={}",
                out[2 * i + 1]
            );
        }
    }

    /// Q5_0 反量化。
    #[test]
    fn test_q5_0() {
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&f16b(0.25)); // d
                                                  // qh: 4 bytes at [2..6]
        block[2] = 0x11;
        block[3] = 0x22;
        block[4] = 0;
        block[5] = 0;
        // qs: 16 bytes at [6..22]
        for (i, b) in block[6..].iter_mut().enumerate() {
            let lo = i as u8;
            let hi = (i as u8).wrapping_add(1) & 0xF;
            *b = lo | (hi << 4);
        }
        let mut out = [0f32; 32];
        dequantize_block(&block, GgmlType::Q5_0, &mut out).unwrap();
        let qh = u32::from_le_bytes([0x11, 0x22, 0, 0]);
        for i in 0usize..16 {
            let lo4 = i as u32;
            let hi4 = (i + 1) as u32 & 0xF;
            let xh_0 = ((qh >> i) << 4) & 0x10; // bit i -> 0x10
            let xh_1 = (qh >> (i + 12)) & 0x10; // bit i+12 -> 0x10
            let lo = lo4 | xh_0;
            let hi = hi4 | xh_1;
            assert!(
                (out[i] - 0.25 * (lo as i32 - 16) as f32).abs() < 1e-6,
                "i={i} lo={lo} out={}",
                out[i]
            );
            assert!(
                (out[16 + i] - 0.25 * (hi as i32 - 16) as f32).abs() < 1e-6,
                "i={i} hi={hi} out={}",
                out[16 + i]
            );
        }
    }

    /// Q5_1 反量化。
    #[test]
    fn test_q5_1() {
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&f16b(0.25)); // d=0.25
        block[2..4].copy_from_slice(&f16b(0.0)); // m=0
                                                 // qh: 4 bytes at [4..8]
        block[4] = 0x0F; // qh bits 0..7
        block[5] = 0x00; // qh bits 8..15
        block[6] = 0;
        block[7] = 0;
        // qs: 16 bytes at [8..24]
        for i in 0..16 {
            let lo = i as u8;
            let hi = (i as u8).wrapping_add(1) & 0xF;
            block[8 + i] = lo | (hi << 4);
        }
        let mut out = [0f32; 32];
        dequantize_block(&block, GgmlType::Q5_1, &mut out).unwrap();
        let qh = u32::from_le_bytes([0x0F, 0x00, 0, 0]);
        for i in 0usize..16 {
            let lo4 = i as u32;
            let hi4 = (i + 1) as u32 & 0xF;
            let xh_0 = ((qh >> i) << 4) & 0x10;
            let xh_1 = (qh >> (i + 12)) & 0x10;
            let lo = lo4 | xh_0;
            let hi = hi4 | xh_1;
            assert!(
                (out[i] - 0.25 * lo as f32).abs() < 1e-6,
                "i={i} lo={lo} out={}",
                out[i]
            );
            assert!(
                (out[16 + i] - 0.25 * hi as f32).abs() < 1e-6,
                "i={i} hi={hi} out={}",
                out[16 + i]
            );
        }
    }

    /// Q8_0 反量化。
    #[test]
    fn test_q8_0() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16b(0.125));
        for (i, b) in block[2..].iter_mut().enumerate() {
            *b = (i % 5) as u8;
        }
        let mut out = [0f32; 32];
        dequantize_block(&block, GgmlType::Q8_0, &mut out).unwrap();
        for i in 0..32 {
            let q = block[2 + i] as i8;
            assert!(
                (out[i] - 0.125 * q as f32).abs() < 1e-6,
                "i={i} q={q} out={}",
                out[i]
            );
        }
    }

    /// K-quant 反量化有限性冒烟（Q2_K~Q8_K）。
    #[test]
    fn test_kquant_finite_smoke() {
        let dtypes = [
            GgmlType::Q2_K,
            GgmlType::Q3_K_S,
            GgmlType::Q3_K_M,
            GgmlType::Q3_K_L,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_K,
        ];
        for dt in dtypes {
            let bb = dt.block_bytes().unwrap() as usize;
            let mut block = vec![0u8; bb];
            // 设置 d = 0.5 在各类型的 d 偏移处
            match dt {
                GgmlType::Q2_K => {
                    block[80..82].copy_from_slice(&f16b(0.5)); // d
                    block[82..84].copy_from_slice(&f16b(0.1)); // dmin
                }
                GgmlType::Q3_K_S | GgmlType::Q3_K_L => {
                    block[108..110].copy_from_slice(&f16b(0.5)); // d
                }
                GgmlType::Q3_K_M => {
                    block[112..114].copy_from_slice(&f16b(0.5)); // d
                }
                GgmlType::Q4_K | GgmlType::Q5_K => {
                    block[0..2].copy_from_slice(&f16b(0.5)); // d
                    block[2..4].copy_from_slice(&f16b(0.1)); // dmin
                }
                GgmlType::Q6_K => {
                    block[208..210].copy_from_slice(&f16b(0.5)); // d
                }
                GgmlType::Q8_K => {
                    block[0..4].copy_from_slice(&0.5f32.to_le_bytes()); // d (f32)
                }
                _ => unreachable!(),
            }
            // 设置一些 scales 为非零值（避免全零导致全零输出）
            for (i, b) in block.iter_mut().enumerate() {
                if *b == 0 {
                    *b = (i as u8) & 0x0F;
                }
            }
            let mut out = vec![0f32; 256];
            dequantize_block(&block, dt, &mut out).unwrap();
            for v in &out {
                assert!(v.is_finite(), "{dt:?} produced non-finite");
            }
        }
    }

    /// Q4_K 反量化与手算对比（d=0.5，dmin=0.1，6-bit scale+min 编码，qs 已知）。
    #[test]
    fn test_q4_k() {
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&f16b(0.5)); // d = 0.5
        block[2..4].copy_from_slice(&f16b(0.1)); // dmin = 0.1
        // 6-bit scale+min: j=0 → scale=scales[0]&0x3F, min=scales[4]&0x3F
        block[4] = 0x02;  // scales[0] = 2 → j=0 scale = 2
        block[8] = 0x03;  // scales[4] = 3 → j=0 min = 3
                          // qs: 每字节 2 个 4-bit，块内偏移 16
        for i in 0..128 {
            block[16 + i] = 0x5A; // lo=0xA, hi=0x5
        }
        let mut out = vec![0f32; 256];
        dequantize_block(&block, GgmlType::Q4_K, &mut out).unwrap();
        // j=0 组 (e=0..31): sub_block=0, s1 = 2*0.5 = 1.0, s2 = 3*dmin(f16≈0.099976) ≈ 0.29993
        // block 全 0x5A：e=0..31 用低 nibble(0xA=10)（is_high=0），e=32..63 用高 nibble(0x5=5)（is_high=1）
        // 故 e=0..15 都应 ≈ s1*10 - s2 = 10 - 0.29993 = 9.70007
        for i in 0..16 {
            assert!(
                (out[i] - 9.7).abs() < 1e-4,
                "j=0 k={i} out={}（block 全 0x5A 低 nibble=10，e=0..31 都应 9.7）",
                out[i]
            );
        }
        // j=1..7 组：根据实际 scales 值计算期望，与 deq_q4_k 使用同一 f16 精度值
        // 填充后 scales = block[4..16] = [0x02, 0x05, 0x06, 0x07, 0x03, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
        let scales = &block[4..16];
        let d = f16_at_test(0.5);
        let dmin = f16_at_test(0.1);
        for j in 1..8 {
            let (sc, mn) = k_scale_min_6bit(scales, j);
            let s1 = sc as f32 * d;
            let s2 = mn as f32 * dmin;
            let base = 32 * j;
            for i in 0..32 {
                let q = if i % 2 == 0 { 0xA } else { 0x5 }; // 0x5A: lo=0xA, hi=0x5
                let expect = s1 * q as f32 - s2;
                assert!(
                    (out[base + i] - expect).abs() < 1e-5,
                    "j={j} k={} out={} expected={expect} (sc={sc} mn={mn})",
                    i, out[base + i]
                );
            }
        }
    }

    /// dequantize 整张量（2 个 Q4_0 block）。
    #[test]
    fn test_dequantize_whole() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16b(1.0));
        let mut data = Vec::new();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);
        let out = dequantize(&data, GgmlType::Q4_0, 64).unwrap();
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// dequantize 字节不足报错。
    #[test]
    fn test_dequantize_insufficient_bytes() {
        let data = [0u8; 8];
        assert!(matches!(
            dequantize(&data, GgmlType::Q4_0, 32),
            Err(GgufError::DequantError { .. })
        ));
    }

    /// 不支持的 dtype 报错。
    #[test]
    fn test_dequantize_unsupported() {
        assert!(matches!(
            dequantize_block(&[0u8; 8], GgmlType::Q4_2, &mut [0f32; 32]),
            Err(GgufError::DequantError { .. })
        ));
    }

    /// quant_matvec 与全量反量化 GEMM 对比（Q4_K 小矩阵）。
    #[test]
    fn test_quant_matvec_matches_dequant() {
        let rows: u64 = 8;
        let cols: u64 = 512; // 2 个 Q4_K super-block
        let bb = GgmlType::Q4_K.block_bytes().unwrap() as usize;
        let bs = GgmlType::Q4_K.block_size().unwrap() as usize;
        let nblocks = (cols / bs as u64) as usize;
        let mut w = Vec::with_capacity((rows as usize) * nblocks * bb);
        for r in 0..rows as usize {
            for _b in 0..nblocks {
                // 构造确定的 block：d=0.5，6-bit scale j=0=(r%5)+1，其余 0，qs 全 0x5A
                let mut block = vec![0u8; bb];
                block[0..2].copy_from_slice(&f16b(0.5)); // d
                block[4] = ((r % 5) + 1) as u8; // scales[0] → j=0 scale (6-bit)
                for i in 0..128 {
                    block[16 + i] = 0x5A;
                }
                w.extend_from_slice(&block);
            }
        }
        let x: Vec<f32> = (0..cols as usize).map(|i| (i % 3) as f32).collect();
        let wd = dequantize(&w, GgmlType::Q4_K, rows * cols).unwrap();
        let mut y_ref = vec![0f32; rows as usize];
        for r in 0..rows as usize {
            let mut acc = 0f32;
            for c in 0..cols as usize {
                acc += wd[r * cols as usize + c] * x[c];
            }
            y_ref[r] = acc;
        }
        let mut y = vec![0f32; rows as usize];
        quant_matvec(&w, GgmlType::Q4_K, rows, cols, &x, &mut y).unwrap();
        for r in 0..rows as usize {
            let diff = (y[r] - y_ref[r]).abs();
            assert!(
                diff < 1e-3 * y_ref[r].abs().max(1.0),
                "row {r}: {} vs {}",
                y[r],
                y_ref[r]
            );
        }
    }

    /// quant_matvec F32 路径。
    #[test]
    fn test_quant_matvec_f32() {
        let rows: u64 = 4;
        let cols: u64 = 8;
        let w: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let x = vec![1.0f32; cols as usize];
        let mut y = vec![0f32; rows as usize];
        quant_matvec(&w, GgmlType::F32, rows, cols, &x, &mut y).unwrap();
        for (r, yv) in y.iter().enumerate() {
            let expect: f32 = (0..cols as usize)
                .map(|c| (r * cols as usize + c) as f32)
                .sum();
            assert!((yv - expect).abs() < 1e-4);
        }
    }

    /// quant_matvec F16 路径。
    #[test]
    fn test_quant_matvec_f16() {
        let rows: u64 = 2;
        let cols: u64 = 4;
        let w: Vec<u8> = (0..rows * cols)
            .flat_map(|i| {
                let b = f16b(i as f32);
                [b[0], b[1]]
            })
            .collect();
        let x = vec![2.0f32; cols as usize];
        let mut y = vec![0f32; rows as usize];
        quant_matvec(&w, GgmlType::F16, rows, cols, &x, &mut y).unwrap();
        for (r, yv) in y.iter().enumerate() {
            let expect: f32 = (0..cols as usize)
                .map(|c| (r * cols as usize + c) as f32 * 2.0)
                .sum();
            assert!((yv - expect).abs() < 1e-4);
        }
    }

    /// quant_matvec 维度不匹配报错。
    #[test]
    fn test_quant_matvec_dim_mismatch() {
        let w = [0u8; 16];
        let x = vec![1.0f32; 16];
        let mut y = vec![0f32; 1];
        assert!(matches!(
            quant_matvec(&w, GgmlType::Q4_0, 1, 32, &x, &mut y),
            Err(GgufError::DequantError { .. })
        ));
    }

}
