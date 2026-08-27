//! KV Cache：解码阶段复用历史 key/value，避免重复计算。
//!
//! 每个层（layer）持有独立的 [`KvCache`]，按 token 顺序追加。
//! 注意力阶段只需取当前已写入的 `seq_len()` 个 token 的 K/V。

/// 单层的 KV 缓存。
///
/// - `k`/`v` 均为行主序 f32，形状 `[seq_len, n_kv_heads, head_dim]`。
///   对于 GQA/MQA，`n_kv_heads` 可能小于注意力头数。
/// - 按 token 追加，`get_k`/`get_v` 返回当前已填充的前缀切片。
#[derive(Debug, Clone)]
pub struct KvCache {
    n_kv_heads: usize,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    /// 创建空缓存。`n_kv_heads`：KV 头数，`head_dim`：每头维度。
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            n_kv_heads,
            head_dim,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    /// KV 头数。
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// 每头维度。
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 已缓存的 token 数。
    pub fn seq_len(&self) -> usize {
        self.k.len() / (self.n_kv_heads * self.head_dim).max(1)
    }

    /// 追加一个 token 的 K/V（各 `n_kv_heads * head_dim` 个 f32）。
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let n = self.n_kv_heads * self.head_dim;
        if k.len() < n || v.len() < n {
            return;
        }
        self.k.extend_from_slice(&k[..n]);
        self.v.extend_from_slice(&v[..n]);
    }

    /// 返回当前前缀的 K（`[seq_len * n_kv_heads * head_dim]`）。
    pub fn get_k(&self) -> &[f32] {
        &self.k
    }

    /// 返回当前前缀的 V（`[seq_len * n_kv_heads * head_dim]`）。
    pub fn get_v(&self) -> &[f32] {
        &self.v
    }

    /// 取第 `seq` 个 token 的某 KV 头 K（`head_dim` 长）。
    pub fn k_at(&self, seq: usize, head: usize) -> Option<&[f32]> {
        let n = self.n_kv_heads * self.head_dim;
        let base = seq * n + head * self.head_dim;
        self.k.get(base..base + self.head_dim)
    }

    /// 取第 `seq` 个 token 的某 KV 头 V（`head_dim` 长）。
    pub fn v_at(&self, seq: usize, head: usize) -> Option<&[f32]> {
        let n = self.n_kv_heads * self.head_dim;
        let base = seq * n + head * self.head_dim;
        self.v.get(base..base + self.head_dim)
    }

    /// 清空缓存（如新对话开始时）。
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }

    /// 截断到前 `new_len` 个 token（如回退采样时）。
    pub fn truncate(&mut self, new_len: usize) {
        let n = self.n_kv_heads * self.head_dim;
        let keep = new_len * n;
        if keep < self.k.len() {
            self.k.truncate(keep);
            self.v.truncate(keep);
        }
    }
}

/// 全部层的 KV 缓存集合，按层索引访问。
#[derive(Debug, Clone)]
pub struct Cache {
    layers: Vec<KvCache>,
}

impl Cache {
    /// 为 `n_layers` 层各创建一个 `n_kv_heads × head_dim` 的缓存。
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| KvCache::new(n_kv_heads, head_dim))
                .collect(),
        }
    }

    /// 层数。
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// 取某层的缓存（只读）。
    pub fn layer(&self, l: usize) -> Option<&KvCache> {
        self.layers.get(l)
    }

    /// 取某层的缓存（可变）。
    pub fn layer_mut(&mut self, l: usize) -> Option<&mut KvCache> {
        self.layers.get_mut(l)
    }

    /// 清空所有层。
    pub fn clear_all(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }

    /// 所有层截断到 `new_len`。
    pub fn truncate_all(&mut self, new_len: usize) {
        for c in &mut self.layers {
            c.truncate(new_len);
        }
    }
}

// ---------- 测试 ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_get() {
        let mut c = KvCache::new(2, 4); // 2 kv heads, head_dim 4
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.get_k().len(), 0);

        // token0: K = 2*4 = 8 floats
        let k0 = vec![0.1f32; 8];
        let v0 = vec![0.2f32; 8];
        c.append(&k0, &v0);
        assert_eq!(c.seq_len(), 1);
        assert_eq!(c.get_k().len(), 8);

        // token1
        let k1 = vec![0.3f32; 8];
        let v1 = vec![0.4f32; 8];
        c.append(&k1, &v1);
        assert_eq!(c.seq_len(), 2);
        assert_eq!(c.get_v().len(), 16);
    }

    #[test]
    fn test_k_at_v_at() {
        let mut c = KvCache::new(2, 2); // 2 kv heads, head_dim 2 → 每 token 4 floats
                                        // seq0: head0=[1,2], head1=[3,4]
        c.append(&[1.0f32, 2.0, 3.0, 4.0], &[10.0f32, 20.0, 30.0, 40.0]);
        let k0 = c.k_at(0, 0).unwrap();
        assert_eq!(k0, &[1.0f32, 2.0]);
        let k1 = c.k_at(0, 1).unwrap();
        assert_eq!(k1, &[3.0f32, 4.0]);
        let v0 = c.v_at(0, 0).unwrap();
        assert_eq!(v0, &[10.0f32, 20.0]);
        // 越界
        assert!(c.k_at(1, 0).is_none());
    }

    #[test]
    fn test_truncate() {
        let mut c = KvCache::new(2, 2);
        for i in 0..4u32 {
            let k: Vec<f32> = vec![i as f32; 4];
            let v: Vec<f32> = vec![(i + 100) as f32; 4];
            c.append(&k, &v);
        }
        assert_eq!(c.seq_len(), 4);
        c.truncate(2);
        assert_eq!(c.seq_len(), 2);
        // 保留前 2 个 token（head 0 各 head_dim=2 个值）
        assert_eq!(c.get_k().len(), 8);
        assert_eq!(c.k_at(0, 0).unwrap(), &[0.0f32, 0.0]);
        assert_eq!(c.k_at(1, 0).unwrap(), &[1.0f32, 1.0]);
        assert!(c.k_at(2, 0).is_none());
    }

    #[test]
    fn test_clear() {
        let mut c = KvCache::new(2, 4);
        c.append(&[0.0f32; 8], &[0.0f32; 8]);
        assert_eq!(c.seq_len(), 1);
        c.clear();
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.get_k().len(), 0);
        assert_eq!(c.get_v().len(), 0);
    }

    #[test]
    fn test_append_short_ignored() {
        let mut c = KvCache::new(2, 4);
        // 不足 8 floats，忽略
        c.append(&[0.0f32; 3], &[0.0f32; 3]);
        assert_eq!(c.seq_len(), 0);
    }

    #[test]
    fn test_cache_layers() {
        let mut cache = Cache::new(3, 2, 4);
        assert_eq!(cache.n_layers(), 3);
        assert!(cache.layer(0).is_some());
        assert!(cache.layer(3).is_none());
        // 每层独立追加
        if let Some(l) = cache.layer_mut(1) {
            l.append(&[0.5f32; 8], &[0.6f32; 8]);
        }
        assert_eq!(cache.layer(0).unwrap().seq_len(), 0);
        assert_eq!(cache.layer(1).unwrap().seq_len(), 1);
        assert_eq!(cache.layer(2).unwrap().seq_len(), 0);
        // truncate_all
        cache.truncate_all(0);
        assert_eq!(cache.layer(1).unwrap().seq_len(), 0);
    }
}
