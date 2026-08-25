use std::fmt;

/// GGUF 解析过程中可能出现的错误。
#[derive(Debug)]
pub enum GgufError {
    /// 文件 I/O 错误
    Io(std::io::Error),
    /// mmap 错误
    Mmap(String),
    /// 魔数不匹配（非 GGUF 文件）
    InvalidMagic(u32),
    /// 不支持的版本
    UnsupportedVersion(u32),
    /// 读取越界（文件损坏/截断）
    OutOfBounds {
        offset: u64,
        required: u64,
        file_size: u64,
    },
    /// 非法的 KV 类型值
    InvalidGgufType(i32),
    /// 数组元素类型非法（如嵌套数组）
    InvalidArrayElemType(i32),
    /// 字符串长度非法（如超过剩余字节）
    InvalidStringLength(u64),
    /// 张量维度非法（负数等）
    InvalidTensorDim { name: String, dim: i64 },
    /// 非法的计数字段（n_kv / n_tensors 为负或超大）
    InvalidCount { field: &'static str, value: i64 },
    /// 其他
    Other(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::Mmap(msg) => write!(f, "mmap error: {msg}"),
            GgufError::InvalidMagic(m) => {
                write!(f, "invalid GGUF magic number: 0x{m:08x} (expected 0x46554747)")
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version: {v} (expected 3)")
            }
            GgufError::OutOfBounds {
                offset,
                required,
                file_size,
            } => write!(
                f,
                "read out of bounds at offset {offset}: need {required} bytes but only {} remain (file size {file_size})",
                file_size.saturating_sub(*offset).min(*required)
            ),
            GgufError::InvalidGgufType(t) => write!(f, "invalid gguf_type value: {t}"),
            GgufError::InvalidArrayElemType(t) => {
                write!(f, "invalid array element type: {t} (nested arrays are not allowed)")
            }
            GgufError::InvalidStringLength(len) => {
                write!(f, "invalid string length: {len} (exceeds remaining bytes or bad UTF-8)")
            }
            GgufError::InvalidTensorDim { name, dim } => {
                write!(f, "tensor '{name}' has invalid dimension: {dim}")
            }
            GgufError::InvalidCount { field, value } => {
                write!(f, "invalid count for field '{field}': {value}")
            }
            GgufError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// 解析结果类型别名。
pub type GgufResult<T> = Result<T, GgufError>;
