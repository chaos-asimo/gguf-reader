use crate::types::GgmlType;

/// 单个张量的元数据描述（不含权重数据体）。
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    /// 张量名称
    pub name: String,
    /// 各维度大小（逻辑顺序，已从文件存储顺序还原）
    pub shape: Vec<u64>,
    /// 数据类型
    pub dtype: GgmlType,
    /// 张量数据在数据体中的字节偏移
    pub offset: u64,
}

impl TensorInfo {
    /// 元素总数 = shape 各维之积。空 shape 返回 1（标量）。
    pub fn num_elements(&self) -> u64 {
        self.shape
            .iter()
            .fold(1u64, |acc, &d| acc.saturating_mul(d))
    }

    /// 估算单个元素的字节数。
    ///
    /// - 浮点/半精度类型（F32/F16/BF16）返回精确值
    /// - Q4_0 等逐元素量化类型返回 0（实际为 block 存储）
    /// - 其余量化类型返回 `None`（无法精确估算）
    pub fn est_element_size(&self) -> Option<u64> {
        self.dtype.element_size()
    }

    /// 估算张量数据字节数。
    ///
    /// 仅对可精确估算的类型（F32/F16/BF16）返回 `Some`，
    /// 量化类型返回 `None`（block 结构复杂，不做估算）。
    pub fn est_data_size(&self) -> Option<u64> {
        if self.dtype.is_floating_point() {
            let elem = self.dtype.element_size()?;
            Some(self.num_elements().saturating_mul(elem))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgmlType;

    #[test]
    fn test_num_elements() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![128, 4096],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 128 * 4096);
    }

    #[test]
    fn test_num_elements_empty_shape() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 1);
    }

    #[test]
    fn test_num_elements_single() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![4096],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 4096);
    }

    #[test]
    fn test_est_data_size_f32() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![4096],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.est_data_size(), Some(4096 * 4));
    }

    #[test]
    fn test_est_data_size_bf16() {
        let t = TensorInfo {
            name: "embd".into(),
            shape: vec![128256, 4096],
            dtype: GgmlType::BF16,
            offset: 0,
        };
        assert_eq!(t.est_data_size(), Some(128256 * 4096 * 2));
    }

    #[test]
    fn test_est_data_size_quantized_none() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![4096, 4096],
            dtype: GgmlType::Q4_K,
            offset: 0,
        };
        assert_eq!(t.est_data_size(), None);
    }

    #[test]
    fn test_est_element_size() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![10],
            dtype: GgmlType::F16,
            offset: 0,
        };
        assert_eq!(t.est_element_size(), Some(2));
    }

    /// 3D 张量元素数 = 各维之积。
    #[test]
    fn test_num_elements_3d() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![2, 3, 4],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 24);
    }

    /// 含 0 维的 shape：元素数为 0（如 [0, 5]）。
    #[test]
    fn test_num_elements_with_zero_dim() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![0, 5],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 0);
        assert_eq!(t.est_data_size(), Some(0));
    }

    /// 超大 shape 防溢出：saturating_mul 应饱和到 u64::MAX 而非 panic。
    #[test]
    fn test_num_elements_overflow_saturates() {
        let t = TensorInfo {
            name: "huge".into(),
            shape: vec![u64::MAX, 2],
            dtype: GgmlType::F32,
            offset: 0,
        };
        // u64::MAX * 2 饱和到 u64::MAX
        assert_eq!(t.num_elements(), u64::MAX);
        // est_data_size: u64::MAX * 4 也饱和到 u64::MAX
        assert_eq!(t.est_data_size(), Some(u64::MAX));
    }

    /// 多个超大维相乘仍饱和（不 panic）。
    #[test]
    fn test_num_elements_multi_huge() {
        let t = TensorInfo {
            name: "h".into(),
            shape: vec![u64::MAX / 2, u64::MAX / 2],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), u64::MAX);
    }

    /// F16 多维张量 est_data_size = 元素数 * 2。
    #[test]
    fn test_est_data_size_f16_multidim() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![10, 20, 30],
            dtype: GgmlType::F16,
            offset: 0,
        };
        assert_eq!(t.est_data_size(), Some(10 * 20 * 30 * 2));
    }

    /// Q4_0 类型：element_size 为 0（block 存储），est_data_size 为 None。
    #[test]
    fn test_q4_0_not_floating_point() {
        let t = TensorInfo {
            name: "t".into(),
            shape: vec![100],
            dtype: GgmlType::Q4_0,
            offset: 0,
        };
        assert_eq!(t.est_element_size(), Some(0));
        assert_eq!(t.est_data_size(), None); // Q4_0 非浮点，不估算
    }
}
