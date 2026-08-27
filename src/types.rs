use crate::error::GgufError;
use std::fmt;

/// GGUF KV 元数据类型（对应 gguf_type 枚举）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    F64 = 12,
}

impl GgufType {
    /// 从原始 int32 值构造，非法值返回 [`GgufError::InvalidGgufType`]。
    pub fn from_i32(v: i32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => GgufType::Uint8,
            1 => GgufType::Int8,
            2 => GgufType::Uint16,
            3 => GgufType::Int16,
            4 => GgufType::Uint32,
            5 => GgufType::Int32,
            6 => GgufType::F32,
            7 => GgufType::Bool,
            8 => GgufType::String,
            9 => GgufType::Array,
            10 => GgufType::Uint64,
            11 => GgufType::Int64,
            12 => GgufType::F64,
            other => return Err(GgufError::InvalidGgufType(other)),
        })
    }

    /// 标量元素的最小字节大小（用于数组预检）。STRING 返回 8（仅长度前缀下限）。
    pub fn min_element_size(self) -> u64 {
        match self {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => 1,
            GgufType::Uint16 | GgufType::Int16 => 2,
            GgufType::Uint32 | GgufType::Int32 | GgufType::F32 => 4,
            GgufType::Uint64 | GgufType::Int64 | GgufType::F64 => 8,
            GgufType::String => 8,
            GgufType::Array => 0,
        }
    }
}

impl fmt::Display for GgufType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GgufType::Uint8 => "uint8",
            GgufType::Int8 => "int8",
            GgufType::Uint16 => "uint16",
            GgufType::Int16 => "int16",
            GgufType::Uint32 => "uint32",
            GgufType::Int32 => "int32",
            GgufType::F32 => "float32",
            GgufType::Bool => "bool",
            GgufType::String => "string",
            GgufType::Array => "array",
            GgufType::Uint64 => "uint64",
            GgufType::Int64 => "int64",
            GgufType::F64 => "float64",
        };
        f.write_str(s)
    }
}

/// 张量数据类型（对应 ggml_type 枚举）。常见值具名，未知值降级为 [`GgmlType::Unknown`]。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K_L = 11,
    Q3_K_M = 12,
    Q3_K_S = 13,
    Q4_K = 14,
    Q5_K = 15,
    Q6_K = 16,
    Q8_K = 17,
    IQ2_XXS = 18,
    IQ2_XS = 19,
    IQ3_XXS = 20,
    IQ1_S = 21,
    IQ4_NL = 22,
    IQ3_S = 23,
    IQ2_S = 24,
    IQ2_M = 25,
    IQ3_M = 26,
    IQ1_M = 27,
    IQ4_XS = 28,
    IQ3_XS = 29,
    BF16 = 30,
    TQ1_0 = 31,
    TQ2_0 = 32,
    MXFP4 = 33,
    Unknown(i32),
}

impl GgmlType {
    /// 从原始 int32 值构造，未知值降级为 [`GgmlType::Unknown`]（不报错）。
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            4 => GgmlType::Q4_2,
            5 => GgmlType::Q4_3,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            // GGUF 只有一个 Q3_K 类型（值 11）。deq_q3_k_s 与 deq_q3_k_l 布局相同，
            // 此处映射到 Q3_K_S；Q3_K_M（值 114B/block）需通过 infer_k_quant_dtype 推断。
            11 => GgmlType::Q3_K_S,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            18 => GgmlType::IQ2_XXS,
            19 => GgmlType::IQ2_XS,
            20 => GgmlType::IQ3_XXS,
            21 => GgmlType::IQ1_S,
            22 => GgmlType::IQ4_NL,
            23 => GgmlType::IQ3_S,
            24 => GgmlType::IQ2_S,
            25 => GgmlType::IQ2_M,
            26 => GgmlType::IQ3_M,
            27 => GgmlType::IQ1_M,
            28 => GgmlType::IQ4_XS,
            29 => GgmlType::IQ3_XS,
            30 => GgmlType::BF16,
            31 => GgmlType::TQ1_0,
            32 => GgmlType::TQ2_0,
            33 => GgmlType::MXFP4,
            other => GgmlType::Unknown(other),
        }
    }

    /// 单个元素的精确字节大小。
    ///
    /// 浮点/半精度类型为精确值；量化类型（K-quants/I-quants）采用 block 存储，
    /// 返回 `None` 表示无法精确估算（CLI 显示为 "—"）。
    pub fn element_size(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::Q4_0 => Some(0), // 量化 block，非逐元素
            _ => None,
        }
    }

    /// 是否属于可精确估算逐元素大小的类型（F32/F16/BF16）。
    pub fn is_floating_point(self) -> bool {
        matches!(self, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
    }

    /// 量化 block 大小（每 block 覆盖的元素数）。
    /// 浮点类型返回 1。未知类型返回 None。
    pub fn block_size(self) -> Option<u64> {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Some(1),
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => {
                Some(32)
            }
            GgmlType::Q2_K
            | GgmlType::Q3_K_S
            | GgmlType::Q3_K_M
            | GgmlType::Q3_K_L
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K => Some(256),
            _ => None,
        }
    }

    /// 单个 block 的字节数。
    /// 浮点类型返回元素字节数。未知类型返回 None。
    pub fn block_bytes(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::Q4_0 => Some(18),
            GgmlType::Q4_1 => Some(20),
            GgmlType::Q5_0 => Some(22),
            GgmlType::Q5_1 => Some(24),
            GgmlType::Q8_0 => Some(34),
            GgmlType::Q2_K => Some(84),
            GgmlType::Q3_K_S | GgmlType::Q3_K_L => Some(110),
            GgmlType::Q3_K_M => Some(114),
            GgmlType::Q4_K => Some(144),
            GgmlType::Q5_K => Some(176),
            GgmlType::Q6_K => Some(210),
            GgmlType::Q8_K => Some(292),
            _ => None,
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q4_2 => "Q4_2",
            GgmlType::Q4_3 => "Q4_3",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q8_1 => "Q8_1",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q3_K_L => "Q3_K_L",
            GgmlType::Q3_K_M => "Q3_K_M",
            GgmlType::Q3_K_S => "Q3_K_S",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::Q8_K => "Q8_K",
            GgmlType::IQ2_XXS => "IQ2_XXS",
            GgmlType::IQ2_XS => "IQ2_XS",
            GgmlType::IQ3_XXS => "IQ3_XXS",
            GgmlType::IQ1_S => "IQ1_S",
            GgmlType::IQ4_NL => "IQ4_NL",
            GgmlType::IQ3_S => "IQ3_S",
            GgmlType::IQ2_S => "IQ2_S",
            GgmlType::IQ2_M => "IQ2_M",
            GgmlType::IQ3_M => "IQ3_M",
            GgmlType::IQ1_M => "IQ1_M",
            GgmlType::IQ4_XS => "IQ4_XS",
            GgmlType::IQ3_XS => "IQ3_XS",
            GgmlType::BF16 => "BF16",
            GgmlType::TQ1_0 => "TQ1_0",
            GgmlType::TQ2_0 => "TQ2_0",
            GgmlType::MXFP4 => "MXFP4",
            GgmlType::Unknown(n) => {
                return write!(f, "UnknownType({n})");
            }
        };
        f.write_str(s)
    }
}

/// 单个 KV 元数据的值（动态类型）。
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// 数组：元素类型 + 元素序列（元素必须为标量，不允许嵌套数组）
    Array(GgufArray),
}

/// 数组类型，元素同质。
#[derive(Clone, Debug, PartialEq)]
pub struct GgufArray {
    /// 元素类型（不含 [`GgufType::Array`]，GGUF 不允许嵌套数组）
    pub elem_type: GgufType,
    /// 元素序列，每个元素为标量 [`GgufValue`]
    pub data: Vec<GgufValue>,
}

impl GgufValue {
    /// 返回值的类型标签。
    pub fn value_type(&self) -> GgufType {
        match self {
            GgufValue::U8(_) => GgufType::Uint8,
            GgufValue::I8(_) => GgufType::Int8,
            GgufValue::U16(_) => GgufType::Uint16,
            GgufValue::I16(_) => GgufType::Int16,
            GgufValue::U32(_) => GgufType::Uint32,
            GgufValue::I32(_) => GgufType::Int32,
            GgufValue::F32(_) => GgufType::F32,
            GgufValue::Bool(_) => GgufType::Bool,
            GgufValue::String(_) => GgufType::String,
            GgufValue::U64(_) => GgufType::Uint64,
            GgufValue::I64(_) => GgufType::Int64,
            GgufValue::F64(_) => GgufType::F64,
            GgufValue::Array(_) => GgufType::Array,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// 提取为有符号 64 位整数（仅对整数变体生效）。
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufValue::I8(v) => Some(*v as i64),
            GgufValue::I16(v) => Some(*v as i64),
            GgufValue::I32(v) => Some(*v as i64),
            GgufValue::I64(v) => Some(*v),
            GgufValue::U8(v) => Some(*v as i64),
            GgufValue::U16(v) => Some(*v as i64),
            GgufValue::U32(v) => Some(*v as i64),
            GgufValue::U64(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// 提取为浮点 64 位（对整数与浮点变体均生效）。
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            GgufValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&GgufArray> {
        match self {
            GgufValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// 人类可读字符串表示（用于 CLI 文本输出）。
    pub fn display(&self) -> String {
        match self {
            GgufValue::String(s) => format!("\"{s}\""),
            GgufValue::Bool(v) => v.to_string(),
            GgufValue::F32(v) => format_finite(*v as f64),
            GgufValue::F64(v) => format_finite(*v),
            GgufValue::Array(a) => {
                let n = a.data.len();
                let preview: Vec<String> = a.data.iter().take(5).map(|v| v.display()).collect();
                if n <= 5 {
                    format!("[{}]", preview.join(", "))
                } else {
                    format!("[{} elements] (first 5: {})", n, preview.join(", "))
                }
            }
            other => format!("{other:?}"),
        }
    }
}

/// 浮点格式化：NaN/Inf 转为字符串，避免显示 "NaN"/"inf" 歧义。
fn format_finite(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else {
        // 整数值的浮点去掉小数尾
        if v.fract() == 0.0 && v.abs() < 1e15 {
            format!("{v:.1}")
        } else {
            format!("{v}")
        }
    }
}

/// 将 [`GgufValue`] 序列化为 serde_json::Value（供 CLI JSON 输出使用）。
///
/// 仅在 `json` feature 开启时可用。
///
/// 大数组截断：当 `max_array_elements` 不为 None 且元素数超过该值时，
/// 仅保留前 N 项并标记 `truncated=true` 与 `total`。
#[cfg(feature = "json")]
pub fn value_to_json(v: &GgufValue, max_array_elements: Option<usize>) -> serde_json::Value {
    use serde_json::{json, Value as J};
    match v {
        GgufValue::U8(x) => J::from(*x),
        GgufValue::I8(x) => J::from(*x),
        GgufValue::U16(x) => J::from(*x),
        GgufValue::I16(x) => J::from(*x),
        GgufValue::U32(x) => J::from(*x),
        GgufValue::I32(x) => J::from(*x),
        GgufValue::U64(x) => J::from(*x),
        GgufValue::I64(x) => J::from(*x),
        GgufValue::F32(x) => json!(x),
        GgufValue::F64(x) => json!(x),
        GgufValue::Bool(x) => J::from(*x),
        GgufValue::String(s) => J::String(s.clone()),
        GgufValue::Array(a) => {
            let total = a.data.len();
            let limit = max_array_elements.unwrap_or(usize::MAX);
            let shown = a
                .data
                .iter()
                .take(limit)
                .map(|e| value_to_json(e, None))
                .collect::<Vec<_>>();
            if total > limit {
                json!({
                    "element_type": a.elem_type.to_string(),
                    "count": total,
                    "value": shown,
                    "truncated": true,
                    "total": total,
                })
            } else {
                json!({
                    "element_type": a.elem_type.to_string(),
                    "count": total,
                    "value": shown,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_type_from_i32() {
        for i in 0..=12 {
            assert!(GgufType::from_i32(i).is_ok());
        }
        assert_eq!(GgufType::from_i32(0).unwrap(), GgufType::Uint8);
        assert_eq!(GgufType::from_i32(9).unwrap(), GgufType::Array);
        assert_eq!(GgufType::from_i32(12).unwrap(), GgufType::F64);
        assert!(matches!(
            GgufType::from_i32(13),
            Err(GgufError::InvalidGgufType(13))
        ));
        assert!(matches!(
            GgufType::from_i32(-1),
            Err(GgufError::InvalidGgufType(-1))
        ));
    }

    #[test]
    fn test_gguf_type_min_size() {
        assert_eq!(GgufType::Uint8.min_element_size(), 1);
        assert_eq!(GgufType::Uint32.min_element_size(), 4);
        assert_eq!(GgufType::Uint64.min_element_size(), 8);
        assert_eq!(GgufType::String.min_element_size(), 8);
    }

    #[test]
    fn test_ggml_type_from_i32() {
        assert_eq!(GgmlType::from_i32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_i32(30), GgmlType::BF16);
        assert_eq!(GgmlType::from_i32(99), GgmlType::Unknown(99));
        assert_eq!(GgmlType::Unknown(99).to_string(), "UnknownType(99)");
    }

    #[test]
    fn test_ggml_type_element_size() {
        assert_eq!(GgmlType::F32.element_size(), Some(4));
        assert_eq!(GgmlType::F16.element_size(), Some(2));
        assert_eq!(GgmlType::BF16.element_size(), Some(2));
        assert_eq!(GgmlType::Q4_0.element_size(), Some(0));
        assert_eq!(GgmlType::Q4_K.element_size(), None);
    }

    #[test]
    fn test_value_type() {
        assert_eq!(GgufValue::U8(1).value_type(), GgufType::Uint8);
        assert_eq!(GgufValue::String("a".into()).value_type(), GgufType::String);
        assert_eq!(
            GgufValue::Array(GgufArray {
                elem_type: GgufType::F32,
                data: vec![]
            })
            .value_type(),
            GgufType::Array
        );
    }

    #[test]
    fn test_as_i64_as_f64() {
        assert_eq!(GgufValue::I32(-5).as_i64(), Some(-5));
        assert_eq!(GgufValue::U32(100).as_i64(), Some(100));
        assert_eq!(GgufValue::I64(-999).as_i64(), Some(-999));
        assert!(GgufValue::U64(u64::MAX).as_i64().is_none());
        assert!(GgufValue::String("x".into()).as_i64().is_none());

        assert_eq!(GgufValue::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(GgufValue::I32(42).as_f64(), Some(42.0));
        assert_eq!(GgufValue::Bool(true).as_f64(), Some(1.0));
    }

    #[test]
    fn test_display() {
        assert_eq!(GgufValue::U32(32).display(), "U32(32)");
        assert_eq!(GgufValue::String("hi".into()).display(), "\"hi\"");
        assert_eq!(GgufValue::Bool(true).display(), "true");
        let arr = GgufValue::Array(GgufArray {
            elem_type: GgufType::String,
            data: vec![
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
            ],
        });
        assert_eq!(arr.display(), r#"["<s>", "</s>"]"#);
        let big = GgufValue::Array(GgufArray {
            elem_type: GgufType::F32,
            data: (0..10).map(|i| GgufValue::F32(i as f32)).collect(),
        });
        assert!(big.display().contains("10 elements"));
    }

    #[test]
    fn test_format_finite() {
        assert_eq!(format_finite(f64::NAN), "NaN");
        assert_eq!(format_finite(f64::INFINITY), "+Inf");
        assert_eq!(format_finite(3.0), "3.0");
        assert_eq!(format_finite(3.5), "3.5");
    }

    /// 13 个标量变体的 value_type 全量核对。
    #[test]
    fn test_value_type_all_variants() {
        assert_eq!(GgufValue::U8(0).value_type(), GgufType::Uint8);
        assert_eq!(GgufValue::I8(0).value_type(), GgufType::Int8);
        assert_eq!(GgufValue::U16(0).value_type(), GgufType::Uint16);
        assert_eq!(GgufValue::I16(0).value_type(), GgufType::Int16);
        assert_eq!(GgufValue::U32(0).value_type(), GgufType::Uint32);
        assert_eq!(GgufValue::I32(0).value_type(), GgufType::Int32);
        assert_eq!(GgufValue::F32(0.0).value_type(), GgufType::F32);
        assert_eq!(GgufValue::Bool(true).value_type(), GgufType::Bool);
        assert_eq!(GgufValue::String("".into()).value_type(), GgufType::String);
        assert_eq!(GgufValue::U64(0).value_type(), GgufType::Uint64);
        assert_eq!(GgufValue::I64(0).value_type(), GgufType::Int64);
        assert_eq!(GgufValue::F64(0.0).value_type(), GgufType::F64);
    }

    /// as_str / as_bool 对非目标变体应返回 None。
    #[test]
    fn test_as_str_as_bool_negative() {
        assert!(GgufValue::U32(1).as_str().is_none());
        assert!(GgufValue::Bool(true).as_str().is_none());
        assert!(GgufValue::U32(1).as_bool().is_none());
        assert!(GgufValue::String("x".into()).as_bool().is_none());
        assert_eq!(GgufValue::Bool(false).as_bool(), Some(false));
        assert_eq!(GgufValue::String("ok".into()).as_str(), Some("ok"));
    }

    /// 浮点 NaN / Inf 在 display 中的字符串表示。
    #[test]
    fn test_display_f32_f64_special() {
        assert_eq!(GgufValue::F32(f32::NAN).display(), "NaN");
        assert_eq!(GgufValue::F32(f32::INFINITY).display(), "+Inf");
        assert_eq!(GgufValue::F32(f32::NEG_INFINITY).display(), "-Inf");
        assert_eq!(GgufValue::F64(f64::NAN).display(), "NaN");
        assert_eq!(GgufValue::F64(1.25).display(), "1.25");
        assert_eq!(GgufValue::F64(2.0).display(), "2.0");
    }

    /// 各标量类型的 Display 文本（用于 CLI 类型列）。
    #[test]
    fn test_gguf_type_display() {
        assert_eq!(GgufType::Uint8.to_string(), "uint8");
        assert_eq!(GgufType::Int64.to_string(), "int64");
        assert_eq!(GgufType::F32.to_string(), "float32");
        assert_eq!(GgufType::F64.to_string(), "float64");
        assert_eq!(GgufType::Bool.to_string(), "bool");
        assert_eq!(GgufType::String.to_string(), "string");
        assert_eq!(GgufType::Array.to_string(), "array");
    }

    /// 未知 ggml 类型的 Display（数值原样输出）。
    #[test]
    fn test_ggml_type_display_unknown() {
        assert_eq!(GgmlType::F32.to_string(), "F32");
        assert_eq!(GgmlType::Q4_K.to_string(), "Q4_K");
        assert_eq!(GgmlType::BF16.to_string(), "BF16");
        assert_eq!(GgmlType::Unknown(-3).to_string(), "UnknownType(-3)");
    }

    /// is_floating_point 仅对 F32/F16/BF16 成立。
    #[test]
    fn test_is_floating_point() {
        assert!(GgmlType::F32.is_floating_point());
        assert!(GgmlType::F16.is_floating_point());
        assert!(GgmlType::BF16.is_floating_point());
        assert!(!GgmlType::Q4_0.is_floating_point());
        assert!(!GgmlType::Q8_K.is_floating_point());
        assert!(!GgmlType::Unknown(99).is_floating_point());
    }

    /// block_size / block_bytes 对所有支持类型返回正确值。
    #[test]
    fn test_ggml_type_block_size_bytes() {
        // 浮点
        assert_eq!(GgmlType::F32.block_size(), Some(1));
        assert_eq!(GgmlType::F32.block_bytes(), Some(4));
        assert_eq!(GgmlType::F16.block_size(), Some(1));
        assert_eq!(GgmlType::F16.block_bytes(), Some(2));
        assert_eq!(GgmlType::BF16.block_size(), Some(1));
        assert_eq!(GgmlType::BF16.block_bytes(), Some(2));
        // Q4_0 ~ Q8_0
        assert_eq!(GgmlType::Q4_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q4_0.block_bytes(), Some(18));
        assert_eq!(GgmlType::Q4_1.block_size(), Some(32));
        assert_eq!(GgmlType::Q4_1.block_bytes(), Some(20));
        assert_eq!(GgmlType::Q5_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q5_0.block_bytes(), Some(22));
        assert_eq!(GgmlType::Q5_1.block_size(), Some(32));
        assert_eq!(GgmlType::Q5_1.block_bytes(), Some(24));
        assert_eq!(GgmlType::Q8_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q8_0.block_bytes(), Some(34));
        // K-quants
        assert_eq!(GgmlType::Q2_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q2_K.block_bytes(), Some(84));
        assert_eq!(GgmlType::Q3_K_S.block_size(), Some(256));
        assert_eq!(GgmlType::Q3_K_S.block_bytes(), Some(110));
        assert_eq!(GgmlType::Q3_K_M.block_size(), Some(256));
        assert_eq!(GgmlType::Q3_K_M.block_bytes(), Some(114));
        assert_eq!(GgmlType::Q3_K_L.block_size(), Some(256));
        assert_eq!(GgmlType::Q3_K_L.block_bytes(), Some(110));
        assert_eq!(GgmlType::Q4_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q4_K.block_bytes(), Some(144));
        assert_eq!(GgmlType::Q5_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q5_K.block_bytes(), Some(176));
        assert_eq!(GgmlType::Q6_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q6_K.block_bytes(), Some(210));
        assert_eq!(GgmlType::Q8_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q8_K.block_bytes(), Some(292));
        // 未知类型
        assert_eq!(GgmlType::Q4_2.block_size(), None);
        assert_eq!(GgmlType::Q4_2.block_bytes(), None);
        assert_eq!(GgmlType::Unknown(99).block_size(), None);
        assert_eq!(GgmlType::Unknown(99).block_bytes(), None);
    }

    /// as_i64 覆盖全部整数变体（含边界值）。
    #[test]
    fn test_as_i64_all_int_variants() {
        assert_eq!(GgufValue::U8(u8::MAX).as_i64(), Some(255));
        assert_eq!(GgufValue::I8(i8::MIN).as_i64(), Some(-128));
        assert_eq!(GgufValue::U16(u16::MAX).as_i64(), Some(65535));
        assert_eq!(GgufValue::I16(i16::MIN).as_i64(), Some(-32768));
        assert_eq!(GgufValue::U32(u32::MAX).as_i64(), Some(4294967295));
        assert_eq!(GgufValue::I32(i32::MIN).as_i64(), Some(-2147483648));
        assert_eq!(GgufValue::U64(0).as_i64(), Some(0));
        assert_eq!(GgufValue::U64(u64::MAX).as_i64(), None); // 超出 i64 范围
        assert_eq!(GgufValue::I64(i64::MAX).as_i64(), Some(i64::MAX));
    }

    #[cfg(feature = "json")]
    mod json_tests {
        use super::*;

        /// 标量值序列化为 JSON（整数保持 number，浮点保持 number）。
        #[test]
        fn test_value_to_json_scalars() {
            use serde_json::json;
            assert_eq!(value_to_json(&GgufValue::U8(7), None), json!(7));
            assert_eq!(value_to_json(&GgufValue::I8(-5), None), json!(-5));
            assert_eq!(value_to_json(&GgufValue::U16(65535), None), json!(65535));
            assert_eq!(value_to_json(&GgufValue::I16(-1234), None), json!(-1234));
            assert_eq!(
                value_to_json(&GgufValue::U32(0xDEADBEEF), None),
                json!(0xDEADBEEFu32)
            );
            assert_eq!(value_to_json(&GgufValue::I32(-99999), None), json!(-99999));
            assert_eq!(value_to_json(&GgufValue::U64(1), None), json!(1));
            assert_eq!(
                value_to_json(&GgufValue::I64(-12345678901234), None),
                json!(-12345678901234i64)
            );
            assert_eq!(value_to_json(&GgufValue::Bool(true), None), json!(true));
            assert_eq!(value_to_json(&GgufValue::Bool(false), None), json!(false));
            assert_eq!(
                value_to_json(&GgufValue::String("hi".into()), None),
                json!("hi")
            );
            // 浮点：JSON number 比较用 as_f64
            let f32 = value_to_json(&GgufValue::F32(1.5), None);
            assert_eq!(f32.as_f64(), Some(1.5));
            let f64 = value_to_json(&GgufValue::F64(2.25), None);
            assert_eq!(f64.as_f64(), Some(2.25));
        }

        /// 小数组（未超阈值）：不截断，无 truncated 字段。
        #[test]
        fn test_value_to_json_small_array() {
            use serde_json::json;
            let arr = GgufValue::Array(GgufArray {
                elem_type: GgufType::Uint32,
                data: vec![GgufValue::U32(1), GgufValue::U32(2), GgufValue::U32(3)],
            });
            let v = value_to_json(&arr, Some(1000));
            assert_eq!(v["element_type"], "uint32");
            assert_eq!(v["count"], 3);
            assert_eq!(v["value"], json!([1, 2, 3]));
            assert!(v.get("truncated").is_none());
        }

        /// 大数组（超阈值）：截断并标记 truncated=true 与 total。
        #[test]
        fn test_value_to_json_truncated_array() {
            let data: Vec<GgufValue> = (0..1500).map(|i| GgufValue::U32(i as u32)).collect();
            let arr = GgufValue::Array(GgufArray {
                elem_type: GgufType::Uint32,
                data,
            });
            let v = value_to_json(&arr, Some(1000));
            assert_eq!(v["count"], 1500);
            assert_eq!(v["truncated"], true);
            assert_eq!(v["total"], 1500);
            let shown = v["value"].as_array().unwrap();
            assert_eq!(shown.len(), 1000);
            assert_eq!(shown[0], serde_json::json!(0));
            assert_eq!(shown[999], serde_json::json!(999));
        }

        /// max_array_elements=None 表示不截断。
        #[test]
        fn test_value_to_json_no_limit() {
            let data: Vec<GgufValue> = (0..50).map(|i| GgufValue::U8(i as u8)).collect();
            let arr = GgufValue::Array(GgufArray {
                elem_type: GgufType::Uint8,
                data,
            });
            let v = value_to_json(&arr, None);
            assert_eq!(v["count"], 50);
            assert_eq!(v["value"].as_array().unwrap().len(), 50);
            assert!(v.get("truncated").is_none());
        }
    }
}
