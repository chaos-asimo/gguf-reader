# -*- coding: utf-8 -*-
"""用 GGUF 反量化权重复算 prompt='你好' 的 prefill logits top-10。

若 GGUF 权重复算 top ≈ F32 参考 top (3837, 271, ...) → 反量化正确，bug 在 Rust 前向。
若 GGUF 权重复算 top ≈ Rust GEN (644, 2154, ...) → 反量化仍有问题。
"""
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "target" / "qwen2.5-0.5b-instruct-q4_k_m-official.gguf"


# ---------- GGUF 解析 ----------

def parse_gguf(path: Path):
    data = path.read_bytes()
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIqq", data, 0)
    assert magic == 0x46554747
    off = 24
    kvs = {}
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", data, off)[0]; off += 8
        key = data[off:off+klen].decode(); off += klen
        t = struct.unpack_from("<i", data, off)[0]; off += 4
        if t == 0:  # u8
            kvs[key] = data[off]; off += 1
        elif t == 1:  # i8
            kvs[key] = struct.unpack_from("<b", data, off)[0]; off += 1
        elif t == 2:  # u16
            kvs[key] = struct.unpack_from("<H", data, off)[0]; off += 2
        elif t == 3:  # i16
            kvs[key] = struct.unpack_from("<h", data, off)[0]; off += 2
        elif t == 4:  # u32
            kvs[key] = struct.unpack_from("<I", data, off)[0]; off += 4
        elif t == 5:  # i32
            kvs[key] = struct.unpack_from("<i", data, off)[0]; off += 4
        elif t == 6:  # f32
            kvs[key] = struct.unpack_from("<f", data, off)[0]; off += 4
        elif t == 7:  # bool
            kvs[key] = bool(data[off]); off += 1
        elif t == 8:  # string
            vlen = struct.unpack_from("<Q", data, off)[0]; off += 8
            kvs[key] = data[off:off+vlen].decode(); off += vlen
        elif t == 9:  # array
            et = struct.unpack_from("<i", data, off)[0]; off += 4
            n = struct.unpack_from("<q", data, off)[0]; off += 8
            if et == 4:
                kvs[key] = list(struct.unpack_from(f"<{n}I", data, off)); off += n*4
            elif et == 5:
                kvs[key] = list(struct.unpack_from(f"<{n}i", data, off)); off += n*4
            elif et == 6:
                kvs[key] = list(struct.unpack_from(f"<{n}f", data, off)); off += n*4
            elif et == 7:
                kvs[key] = [bool(b) for b in data[off:off+n]]; off += n
            elif et == 8:
                arr = []
                for _ in range(n):
                    s = struct.unpack_from("<Q", data, off)[0]; off += 8
                    arr.append(data[off:off+s].decode()); off += s
                kvs[key] = arr
            else:
                raise ValueError(f"array elem {et} for {key}")
        else:
            raise ValueError(t)
    # tensor infos
    tensors = {}
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", data, off)[0]; off += 8
        name = data[off:off+nlen].decode(); off += nlen
        ndim = struct.unpack_from("<I", data, off)[0]; off += 4
        if ndim > 8:
            raise ValueError(f"bad ndim {ndim} for {name} at off {off-4}")
        dims = list(struct.unpack_from(f"<{ndim}Q", data, off)); off += ndim*8
        dtype = struct.unpack_from("<I", data, off)[0]; off += 4
        t_off = struct.unpack_from("<Q", data, off)[0]; off += 8
        tensors[name] = (dims, dtype, t_off)
    data_start = off
    data_start = (data_start + 31) & ~31
    return kvs, tensors, data, data_start


def align(off, al):
    return (off + al - 1) & ~(al - 1)


# ---------- 反量化 ----------

# ggml_type 原始数值（标准 ggml-quant / GGUF 规范）
# 注意：与 Rust 源码内部 GgmlType 枚举不同，这里用的是文件里存的实际数值
DT_F32 = 0
DT_F16 = 1
DT_Q4_0 = 2
DT_Q4_1 = 3
DT_Q5_0 = 6
DT_Q5_1 = 7
DT_Q8_0 = 8
DT_Q2_K = 10
DT_Q3_K = 11  # 实际类型用 infer 推断（84/110/114 bytes）
DT_Q4_K = 12
DT_Q5_K = 13
DT_Q6_K = 14
DT_Q8_K = 15

def f16_at(b, i):
    return struct.unpack_from("<e", b, i)[0]


def k_scale_min_6bit(scales, j):
    # 参考 llama.cpp get_scale_min_k4 / Rust k_scale_min_6bit：
    # j < 4:  scale = s[j] & 0x3F,        min = s[j+4] & 0x3F
    # j >= 4: scale = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)
    #         min   = (s[j+4] >> 4) | ((s[j] >> 6) << 4)
    if j < 4:
        return (scales[j] & 0x3F), (scales[j+4] & 0x3F)
    sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    mn = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mn


def deq_q4_0(data):
    n = len(data) // 6
    out = np.zeros(n*32, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*6:(bi+1)*6]
        d = f16_at(blk, 0)
        base = bi*32
        for i in range(16):
            b = blk[2+i]
            out[base+i] = d*(((b & 0xF) << 4) - 8)
            out[base+i+16] = d*((b >> 4) - 8)
    return out


def deq_q5_0(data):
    n = len(data) // 22
    out = np.zeros(n*32, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*22:(bi+1)*22]
        d = f16_at(blk, 0)
        qh = struct.unpack_from("<I", blk, 2)[0]
        base = bi*32
        for i in range(16):
            byte = blk[6+i]
            xh0 = ((qh >> i) << 4) & 0x10
            xh1 = (qh >> (i+12)) & 0x10
            out[base+i] = d*(((byte & 0xF) | xh0) - 16)
            out[base+16+i] = d*(((byte >> 4) | xh1) - 16)
    return out


def deq_q8_0(data):
    n = len(data) // 34
    out = np.zeros(n*32, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*34:(bi+1)*34]
        d = f16_at(blk, 0)
        base = bi*32
        for i in range(32):
            v = blk[2+i]
            if v > 127: v -= 256
            out[base+i] = d * v
    return out


def deq_q4_k(data):
    # Q4_K: 144 bytes/256 elems
    # 布局: d f16@0, dmin f16@2, scales[12]@4, qs[128]@16
    # scales: 8 个 6-bit scale + 8 个 6-bit min 交织编码在 12 字节 (get_scale_min_k4)
    # 正确布局（kekzl/imp PR#255 CPU oracle）：
    #   4 组 × 64 元素。对元素 e ∈ [0,256):
    #     group = e >> 6 (0-3), in_grp = e & 63, is_high = in_grp >> 5 (0/1)
    #     byte_in_qs = group*32 + (in_grp & 31)
    #     sub_block = group*2 + is_high
    #     nibble = qs[byte_in_qs] >> 4 if is_high else qs[byte_in_qs] & 0x0F
    n = len(data) // 144
    out = np.zeros(n*256, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*144:(bi+1)*144]
        d = f16_at(blk, 0)
        dmin = f16_at(blk, 2)
        base = bi*256
        for e in range(256):
            group = e >> 6
            in_grp = e & 63
            is_high = in_grp >> 5
            byte_in_qs = group * 32 + (in_grp & 31)
            sub_block = group * 2 + is_high
            sc, mn = k_scale_min_6bit(blk[4:16], sub_block)
            s1 = sc * d
            s2 = mn * dmin
            byte = blk[16 + byte_in_qs]
            val = (byte >> 4) if is_high else (byte & 0x0F)
            out[base + e] = s1 * val - s2
    return out


def deq_q6_k(data):
    # Q6_K: 210 bytes = ql[128]@0 + qh[64]@128 + scales[16 int8]@192 + d(f16)@208
    # 实际文件布局 = Rust deq_q6_k (src/infer/quant.rs)：2 个 128 半区 (n2=0,1)
    #   ql0=blk[n2*64+l] 低4位->q1(元素 l), 高4位->q3(元素 l+64)
    #   ql1=blk[n2*64+32+l] 低4位->q2(元素 l+32), 高4位->q4(元素 l+96)
    #   qh=blk[128+n2*32+l] 4 个 2-bit (shift 0/2/4/6) 分别给 q1/q2/q3/q4
    #   sc 步长 2: sc[ss]->q1, sc[ss+2]->q2, sc[ss+4]->q3, sc[ss+6]->q4, ss=l//16
    #   out = d * s * (q - 32)
    n = len(data) // 210
    out = np.zeros(n*256, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*210:(bi+1)*210]
        d = f16_at(blk, 208)
        base = bi*256
        for n2 in range(2):
            for l in range(32):
                ss = l // 16
                ql0 = blk[n2*64+l]; ql1 = blk[n2*64+32+l]; qhb = blk[128+n2*32+l]
                q1 = (ql0 & 0xF) | (((qhb >> 0) & 3) << 4)
                q2 = (ql1 & 0xF) | (((qhb >> 2) & 3) << 4)
                q3 = (ql0 >> 4) | (((qhb >> 4) & 3) << 4)
                q4 = (ql1 >> 4) | (((qhb >> 6) & 3) << 4)
                s1 = blk[192+n2*8+ss] & 0xFF; s1 -= 256 if s1 > 127 else 0
                s2 = blk[192+n2*8+ss+2] & 0xFF; s2 -= 256 if s2 > 127 else 0
                s3 = blk[192+n2*8+ss+4] & 0xFF; s3 -= 256 if s3 > 127 else 0
                s4 = blk[192+n2*8+ss+6] & 0xFF; s4 -= 256 if s4 > 127 else 0
                h = base + n2*128
                out[h+l] = d*s1*(q1-32)
                out[h+l+32] = d*s2*(q2-32)
                out[h+l+64] = d*s3*(q3-32)
                out[h+l+96] = d*s4*(q4-32)
    return out


def deq_q8_k(data):
    # Q8_K: 292 bytes = d(f32)@0 + qs[256]@4 + bsums[16]@260
    n = len(data) // 292
    out = np.zeros(n*256, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*292:(bi+1)*292]
        d = struct.unpack_from("<f", blk, 0)[0]
        base = bi*256
        qs = np.frombuffer(blk[4:260], dtype=np.int8).astype(np.float32)
        out[base:base+256] = d * qs
    return out


def deq_q5_k(data):
    # Q5_K: 176 bytes = d@0(dmin@2) + scales[12]@4 + qh[32]@16 + qs[128]@48
    n = len(data) // 176
    out = np.zeros(n*256, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*176:(bi+1)*176]
        d = f16_at(blk, 0)
        dmin = f16_at(blk, 2)
        base = bi*256
        for j in range(8):
            sc, mn = k_scale_min_6bit(blk[4:16], j)
            s1 = sc * d
            s2 = mn * dmin
            for i in range(32):
                k = 32*j + i
                byte = blk[48 + (k >> 1)]
                qh_bit = (blk[16 + ((k >> 1) >> 3)] >> ((k >> 1) % 8)) & 1
                val = ((byte & 0xF) if (k & 1) == 0 else (byte >> 4)) | (qh_bit << 4)
                out[base+k] = s1 * val - s2
    return out


DEQ = {
    DT_Q4_0: deq_q4_0,
    DT_Q5_0: deq_q5_0,
    DT_Q8_0: deq_q8_0,
    DT_Q4_K: deq_q4_k,
    DT_Q5_K: deq_q5_k,
    DT_Q6_K: deq_q6_k,
    DT_Q8_K: deq_q8_k,
}


DTYPE_BLOCKS = {
    DT_Q4_0: (32, 6),
    DT_Q4_1: (32, 8),
    DT_Q5_0: (32, 22),
    DT_Q5_1: (32, 24),
    DT_Q8_0: (32, 34),
    DT_Q2_K: (256, 84),
    DT_Q3_K: (256, 110),  # S/L; M 用 114，靠 infer 推断
    DT_Q4_K: (256, 144),
    DT_Q5_K: (256, 176),
    DT_Q6_K: (256, 210),
    DT_Q8_K: (256, 292),
}


def deq_q5_1(data):
    # Q5_1: 24 bytes = d@0 m@2 qh@4 qs@8
    n = len(data) // 24
    out = np.zeros(n*32, dtype=np.float32)
    for bi in range(n):
        blk = data[bi*24:(bi+1)*24]
        d = f16_at(blk, 0)
        m = f16_at(blk, 2)
        qh = struct.unpack_from("<I", blk, 4)[0]
        base = bi*32
        for i in range(16):
            byte = blk[8+i]
            xh0 = ((qh >> i) << 4) & 0x10
            xh1 = (qh >> (i+12)) & 0x10
            out[base+i] = d*(((byte & 0xF) | xh0)) + m
            out[base+16+i] = d*(((byte >> 4) | xh1)) + m
    return out


def infer_k_quant_dtype(name, dtype, raw):
    """GGUF 仅以 Q3_K(11) 区分 K-quant，按每 256 元素字节数推断实际类型。"""
    if dtype == DT_Q2_K:
        return DT_Q2_K
    if dtype == DT_Q3_K:
        per256 = len(raw) / 256
        if abs(per256 - 114) < 1:
            return 114  # Q3_K_M
        return DT_Q3_K
    return dtype


def get_tensor(name):
    dims, dtype, rec_off = TENSORS[name]
    off = DATA_START + rec_off
    nelems = 1
    for d in dims:
        nelems *= d
    if dtype == DT_F32:
        return bytes(DATA[off:off+nelems*4]), dtype
    db, bb = DTYPE_BLOCKS[dtype]
    nblk = nelems // db
    return bytes(DATA[off:off+nblk*bb]), dtype


def dequantize(raw, dtype):
    if dtype == DT_F32:
        return np.frombuffer(np.frombuffer(raw, dtype=np.uint8).tobytes(), dtype=np.float32).copy()
    fn = DEQ.get(dtype)
    if fn is None:
        raise ValueError(f"dtype {dtype} not implemented")
    return fn(raw)


# ---------- 前向 ----------

def matvec_colmajor(a, rows, cols, x):
    # y[i] = sum_j a[j*rows+i]*x[j]
    A = np.ascontiguousarray(a).reshape(cols, rows).T  # [rows, cols]
    return A @ x


def matvec_colmajor_trans(a, dim0, dim1, x):
    # y[i] = sum_j a[j+i*dim0]*x[j]
    B = np.ascontiguousarray(a).reshape(dim1, dim0)  # [dim1, dim0]
    return B @ x


def rmsnorm(x, w, eps=1e-6):
    var = np.sum(np.square(x, dtype=np.float64)) / len(x)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_all(x, positions, inv_freq):
    # x: [n*head_dim], Qwen2 用半分区配对 (i, i+half)（transformers rotate_half 约定），n = len(x)//hd
    # positions: 每个 head/token 组的位置
    half = len(inv_freq)
    hd = half*2
    n = len(x) // hd
    out = x.copy()
    for tok in range(n):
        pos = positions[tok] if tok < len(positions) else 0
        base = tok*hd
        for i in range(half):
            ang = pos * inv_freq[i]
            a, b = x[base+i], x[base+i+half]
            out[base+i] = a*np.cos(ang) - b*np.sin(ang)
            out[base+i+half] = b*np.cos(ang) + a*np.sin(ang)
    return out


def main():
    global DATA, TENSORS, DATA_START
    kvs, tensors, data, data_start = parse_gguf(GGUF)
    DATA, TENSORS, DATA_START = data, tensors, data_start

    d = kvs["qwen2.embedding_length"]
    n_layers = kvs["qwen2.block_count"]
    vocab = 151936
    q = kvs["qwen2.attention.head_count"]
    kv = kvs["qwen2.attention.head_count_kv"]
    f = kvs.get("qwen2.feed_forward_length", kvs.get("qwen2.ffn_length", 4864))
    rope_base = kvs.get("qwen2.rope.freq_base", kvs.get("qwen2.attention.rope.freq_base", 10000.0))
    hd = d // q
    inv_freq = np.array([rope_base ** (-2.0*i/hd) for i in range(hd//2)], dtype=np.float32)
    print(f"d={d} layers={n_layers} vocab={vocab} q={q} kv={kv} f={f} hd={hd}")

    # 预缓存所有层权重（避免重复反量化）
    print("Caching layer weights...")
    layer_cache = {}
    for l in range(n_layers):
        c = {}
        c["attn_n"] = np.frombuffer(get_tensor(f"blk.{l}.attn_norm.weight")[0], dtype=np.uint8).tobytes()
        c["attn_n"] = np.frombuffer(c["attn_n"], dtype=np.float32).copy()
        c["ffn_n"] = np.frombuffer(get_tensor(f"blk.{l}.ffn_norm.weight")[0], dtype=np.uint8).tobytes()
        c["ffn_n"] = np.frombuffer(c["ffn_n"], dtype=np.float32).copy()
        wq_raw, dt_q = get_tensor(f"blk.{l}.attn_q.weight")
        c["wq"] = dequantize(wq_raw, dt_q)
        wk_raw, dt_k = get_tensor(f"blk.{l}.attn_k.weight")
        c["wk"] = dequantize(wk_raw, dt_k)
        wv_raw, dt_v = get_tensor(f"blk.{l}.attn_v.weight")
        c["wv"] = dequantize(wv_raw, dt_v)
        wo_raw, dt_o = get_tensor(f"blk.{l}.attn_output.weight")
        c["wo"] = dequantize(wo_raw, dt_o)
        w1_raw, dt1 = get_tensor(f"blk.{l}.ffn_up.weight")
        c["w1"] = dequantize(w1_raw, dt1)
        w2_raw, dt2 = get_tensor(f"blk.{l}.ffn_gate.weight")
        c["w2"] = dequantize(w2_raw, dt2)
        w3_raw, dt3 = get_tensor(f"blk.{l}.ffn_down.weight")
        c["w3"] = dequantize(w3_raw, dt3)
        # Qwen2 有 attn bias（GGUF bias 与 HF 完全一致，非损坏），必须注入。
        bq_raw, dt_bq = get_tensor(f"blk.{l}.attn_q.bias")
        c["b_q"] = dequantize(bq_raw, dt_bq)
        bk_raw, dt_bk = get_tensor(f"blk.{l}.attn_k.bias")
        c["b_k"] = dequantize(bk_raw, dt_bk)
        bv_raw, dt_bv = get_tensor(f"blk.{l}.attn_v.bias")
        c["b_v"] = dequantize(bv_raw, dt_bv)
        layer_cache[l] = c
    # output norm + lm head
    on_raw = get_tensor("output_norm.weight")[0]
    on_raw = np.frombuffer(on_raw, dtype=np.uint8).tobytes()
    out_n = np.frombuffer(on_raw, dtype=np.float32).copy()
    lm_raw, dt_lm = get_tensor("output.weight")
    lm = dequantize(lm_raw, dt_lm)
    Lm = np.ascontiguousarray(lm).reshape(vocab, d)  # [vocab, d]
    print("Weights cached.")

    # embed
    emb_raw, dt_emb = get_tensor("token_embd.weight")
    emb = dequantize(emb_raw, dt_emb)
    print(f"embed dequant done, mean={emb.mean():.4f} std={emb.std():.4f}")

    # 与 Rust diag-gen 相同的 prompt
    PROMPT_IDS = [14880, 11622, 104811, 102104, 5122, 106582, 104455, 11319]
    n_tok = len(PROMPT_IDS)
    inv = 1.0 / np.sqrt(hd)
    qpk = q // kv

    def forward_batch(tokens, positions):
        """完整 prefill 前向，返回最后一个 token 的 logits。"""
        nt = len(tokens)
        # embed: [nt, d]
        x = np.zeros((nt, d), dtype=np.float32)
        for i, t in enumerate(tokens):
            x[i] = emb[t*d:(t+1)*d]
        for l in range(n_layers):
            c = layer_cache[l]
            # rmsnorm (per token)
            attn_in = np.zeros_like(x)
            for i in range(nt):
                attn_in[i] = rmsnorm(x[i], c["attn_n"])
            # Q [nt, q*hd]
            qv = np.zeros((nt, q*hd), dtype=np.float32)
            for i in range(nt):
                qv[i] = matvec_colmajor_trans(c["wq"], d, q*hd, attn_in[i])
            if c["b_q"] is not None:
                qv += c["b_q"]
            qv = rope_all(qv.flatten(), np.repeat(positions, q), inv_freq).reshape(nt, q*hd)
            # K [nt, kv*hd]
            kb = np.zeros((nt, kv*hd), dtype=np.float32)
            for i in range(nt):
                kb[i] = matvec_colmajor_trans(c["wk"], d, kv*hd, attn_in[i])
            if c["b_k"] is not None:
                kb += c["b_k"]
            kb = rope_all(kb.flatten(), np.repeat(positions, kv), inv_freq).reshape(nt, kv*hd)
            # V [nt, kv*hd]
            vb = np.zeros((nt, kv*hd), dtype=np.float32)
            for i in range(nt):
                vb[i] = matvec_colmajor_trans(c["wv"], d, kv*hd, attn_in[i])
            if c["b_v"] is not None:
                vb += c["b_v"]
            # attend
            o = np.zeros((nt, q*hd), dtype=np.float32)
            for i in range(nt):
                for h in range(q):
                    khead = h // qpk
                    qh = qv[i, h*hd:(h+1)*hd]
                    # causal mask: only attend to positions j <= i
                    scores = np.full(nt, -np.inf, dtype=np.float32)
                    for j in range(i+1):
                        kj = kb[j, khead*hd:(khead+1)*hd]
                        scores[j] = np.dot(qh, kj) * inv
                    m = np.max(scores)
                    probs = np.exp(scores - m)
                    probs /= np.sum(probs)
                    oh = np.zeros(hd, dtype=np.float32)
                    for j in range(nt):
                        if probs[j] > 1e-9:
                            oh += probs[j] * vb[j, khead*hd:(khead+1)*hd]
                    o[i, h*hd:(h+1)*hd] = oh
            # O proj
            attn_out = np.zeros((nt, d), dtype=np.float32)
            for i in range(nt):
                attn_out[i] = matvec_colmajor_trans(c["wo"], q*hd, d, o[i])
            x = x + attn_out
            # ffn
            ffn_in = np.zeros_like(x)
            for i in range(nt):
                ffn_in[i] = rmsnorm(x[i], c["ffn_n"])
            ffn_out = np.zeros((nt, d), dtype=np.float32)
            for i in range(nt):
                up = matvec_colmajor_trans(c["w1"], d, f, ffn_in[i])
                gate = matvec_colmajor_trans(c["w2"], d, f, ffn_in[i])
                h = silu(gate) * up
                ffn_out[i] = matvec_colmajor_trans(c["w3"], f, d, h)
            x = x + ffn_out
            if l % 4 == 0:
                print(f"  layer {l}: x mean={x.mean():.4f} std={x.std():.4f}")
        # final norm + lm_head (last token)
        xl = rmsnorm(x[-1], out_n)
        logits = Lm @ xl
        return logits

    print(f"Forward pass with {n_tok} tokens...")
    logits = forward_batch(PROMPT_IDS, list(range(n_tok)))
    top = np.argsort(logits)[::-1][:10]
    print("GGUF-dequant top-10:", [(int(t), round(float(logits[t]), 3)) for t in top])

    PY_REF_GEN = [220, 104455, 9909, 9286, 16488, 21392, 3837, 102500, 15469, 7552]
    RUST_GEN = [86008, 64493, 644, 118124, 37205, 380, 55502, 81456, 12559, 49082]
    print("Python ref GEN first 10:", PY_REF_GEN)
    print("Rust GEN first 10:", RUST_GEN)
    ov_py = set(int(t) for t in top) & set(PY_REF_GEN)
    ov_rust = set(int(t) for t in top) & set(RUST_GEN)
    print(f"overlap with PY ref top: {len(ov_py)}  with RUST top: {len(ov_rust)}")
    # dump logits for comparison
    np.savez(ROOT / "target" / "gguf_fwd_logits.npz",
             logits=logits.astype(np.float32),
             top=top.astype(np.int32))
    print(f"Saved logits to {ROOT/'target'/'gguf_fwd_logits.npz'}")


if __name__ == "__main__":
    main()
