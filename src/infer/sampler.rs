//! LLM 输出采样器。
//!
//! 提供温度缩放、Top-K、Top-P (nucleus)、Min-P 以及贪心 / 随机采样策略。

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// 采样配置。
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 温度（0 = 贪心，>0 = 缩放 logits）
    pub temperature: f32,
    /// Top-K 候选数（0 = 禁用）
    pub top_k: usize,
    /// Top-P 累积概率阈值（1.0 = 禁用）
    pub top_p: f32,
    /// Min-P 相对概率阈值（0.0 = 禁用）
    pub min_p: f32,
    /// 重复惩罚（1.0 = 禁用，>1.0 惩罚已出现 token）
    pub repeat_penalty: f32,
    /// 种子（0 = 使用系统随机）
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }
}

/// 采样器：持有配置与 RNG 状态。
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
    /// 已出现 token 计数（用于 repeat penalty）
    seen: std::collections::HashMap<u32, f32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = if config.seed != 0 {
            StdRng::seed_from_u64(config.seed)
        } else {
            StdRng::from_entropy()
        };
        Self {
            config,
            rng,
            seen: std::collections::HashMap::new(),
        }
    }

    /// 配置只读访问。
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// 更新采样配置（seed 变更时重置 RNG）。
    /// GUI 中参数实时修改时调用，无需重建 Sampler。
    pub fn set_config(&mut self, config: SamplerConfig) {
        if config.seed != self.config.seed && config.seed != 0 {
            self.rng = StdRng::seed_from_u64(config.seed);
        }
        self.config = config;
    }

    /// 清空 repeat penalty 计数与 RNG 状态（新对话时调用）。
    pub fn reset(&mut self) {
        self.seen.clear();
    }

    /// 记录一个已生成 token（更新 repeat penalty 计数）。
    pub fn record(&mut self, token_id: u32) {
        *self.seen.entry(token_id).or_insert(0.0) += 1.0;
    }

    /// 从 logits 中采样一个 token id。
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let vocab = logits.len();
        if vocab == 0 {
            panic!("logits 为空");
        }

        // Step 1: 重复惩罚
        let penalized = if self.config.repeat_penalty != 1.0 {
            let mut v = logits.to_vec();
            for (i, &p) in self.seen.iter() {
                let idx = *i as usize;
                if idx < vocab && p > 0.0 {
                    let penalty = self.config.repeat_penalty;
                    if v[idx] > 0.0 {
                        v[idx] /= penalty;
                    } else {
                        v[idx] *= penalty;
                    }
                }
            }
            v
        } else {
            logits.to_vec()
        };

        // Step 2: 温度缩放
        let temp = self.config.temperature;
        let scaled = if temp > 1e-6 {
            penalized.iter().map(|&x| x / temp).collect::<Vec<f32>>()
        } else {
            // 贪心
            let max = penalized
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            return max;
        };

        // Step 3: Softmax
        let max_v = scaled.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        // Step 4: Top-K
        let mut candidates: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        if self.config.top_k > 0 && self.config.top_k < candidates.len() {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(self.config.top_k);
        }

        // Step 5: Top-P (nucleus)
        if self.config.top_p < 1.0 {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let total: f32 = candidates.iter().map(|(_, p)| *p).sum();
            let threshold = total * self.config.top_p;
            let mut cum = 0.0f32;
            let mut cut = candidates.len();
            for (i, (_, p)) in candidates.iter().enumerate() {
                cum += p;
                if cum >= threshold {
                    cut = i + 1;
                    break;
                }
            }
            candidates.truncate(cut);
        }

        // Step 6: Min-P
        if self.config.min_p > 0.0 {
            let max_prob = candidates.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
            let min_threshold = max_prob * self.config.min_p;
            candidates.retain(|(_, p)| *p >= min_threshold);
        }

        // Step 7: 重归一化并随机选择
        if candidates.is_empty() {
            // 兜底：取原始 logits argmax
            return scaled
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        let total: f32 = candidates.iter().map(|(_, p)| *p).sum();
        let mut r = self.rng.gen_range(0.0..1.0) * total;
        for &(i, p) in &candidates {
            r -= p;
            if r <= 0.0 {
                return i as u32;
            }
        }
        // 浮点误差兜底
        candidates.last().unwrap().0 as u32
    }

    /// 贪心采样（argmax）。
    pub fn greedy(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// 纯随机采样（无过滤）。
    pub fn random(&mut self, probs: &[f32]) -> u32 {
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let mut r = self.rng.gen_range(0.0..total);
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(temp: f32, top_k: usize, top_p: f32) -> SamplerConfig {
        SamplerConfig {
            temperature: temp,
            top_k,
            top_p,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 42,
        }
    }

    #[test]
    fn test_greedy() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        assert_eq!(Sampler::greedy(&logits), 1);
    }

    #[test]
    fn test_temperature_zero_greedy() {
        let cfg = config(0.0, 40, 0.95);
        let mut s = Sampler::new(cfg);
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn test_top_k() {
        // 只有 top-2 候选：logit[1]=10, logit[0]=9, 其余极低
        let cfg = config(1.0, 2, 1.0);
        let mut s = Sampler::new(cfg);
        let logits = vec![9.0f32, 10.0, -100.0, -100.0];
        let id = s.sample(&logits);
        assert!(id == 0 || id == 1);
    }

    #[test]
    fn test_top_p() {
        // top_p=0.5 应排除低概率 token
        let cfg = config(1.0, 100, 0.5);
        let mut s = Sampler::new(cfg);
        let logits = vec![10.0f32, 9.0, -100.0, -100.0];
        let id = s.sample(&logits);
        assert!(id == 0 || id == 1);
    }

    #[test]
    fn test_repeat_penalty() {
        let cfg = SamplerConfig {
            temperature: 0.0, // 贪心
            top_k: 40,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 10.0,
            seed: 42,
        };
        let mut s = Sampler::new(cfg);
        let logits = vec![5.0f32, 5.0]; // 两个并列
                                        // 记录 token 0，再次采样应偏向 token 1
        s.record(0);
        let id = s.sample(&logits);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_min_p() {
        // min_p=0.5：max_prob 的 50%
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 100,
            top_p: 1.0,
            min_p: 0.5,
            repeat_penalty: 1.0,
            seed: 42,
        };
        let mut s = Sampler::new(cfg);
        // probs ≈ [0.9996, 0.0003, ...] → min_p 过滤掉低概率
        let logits = vec![10.0f32, 0.0, -100.0];
        let id = s.sample(&logits);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let cfg = config(1.0, 40, 0.95);
        let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();

        let mut s1 = Sampler::new(cfg.clone());
        let mut s2 = Sampler::new(cfg);
        let a = s1.sample(&logits);
        let b = s2.sample(&logits);
        assert_eq!(a, b);
    }

    #[test]
    fn test_sample_valid_range() {
        let cfg = config(0.8, 20, 0.9);
        let mut s = Sampler::new(cfg);
        let logits: Vec<f32> = (0..50).map(|i| (i as f32) * 0.5 - 12.0).collect();
        for _ in 0..20 {
            let id = s.sample(&logits);
            assert!(id < 50);
            s.record(id);
        }
    }
}
