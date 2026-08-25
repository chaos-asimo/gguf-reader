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
}
