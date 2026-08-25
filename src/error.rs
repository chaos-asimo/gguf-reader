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

#[cfg(test)]
mod tests {
    use super::*;

    /// 各错误变体的 Display 文本包含关键信息。
    #[test]
    fn test_display_messages() {
        let e = GgufError::InvalidMagic(0x12345678);
        assert!(e.to_string().contains("0x12345678"));
        assert!(e.to_string().contains("0x46554747"));

        let e = GgufError::UnsupportedVersion(7);
        assert!(e.to_string().contains("7"));
        assert!(e.to_string().contains("expected 3"));

        let e = GgufError::OutOfBounds {
            offset: 10,
            required: 8,
            file_size: 16,
        };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("8"));
        assert!(e.to_string().contains("16"));

        let e = GgufError::InvalidGgufType(99);
        assert!(e.to_string().contains("99"));

        let e = GgufError::InvalidArrayElemType(9);
        assert!(e.to_string().contains("9"));
        assert!(e.to_string().contains("nested"));

        let e = GgufError::InvalidStringLength(42);
        assert!(e.to_string().contains("42"));

        let e = GgufError::InvalidTensorDim {
            name: "tok".into(),
            dim: -3,
        };
        assert!(e.to_string().contains("tok"));
        assert!(e.to_string().contains("-3"));

        let e = GgufError::InvalidCount {
            field: "n_kv",
            value: -1,
        };
        assert!(e.to_string().contains("n_kv"));
        assert!(e.to_string().contains("-1"));

        let e = GgufError::Mmap("boom".into());
        assert!(e.to_string().contains("boom"));

        let e = GgufError::Other("custom".into());
        assert_eq!(e.to_string(), "custom");
    }

    /// Io 错误实现 std::error::Error 的 source()，其余变体返回 None。
    #[test]
    fn test_error_source() {
        use std::error::Error as _;
        let io = GgufError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "no file"));
        assert!(io.source().is_some());

        let magic = GgufError::InvalidMagic(0);
        assert!(magic.source().is_none());

        let oob = GgufError::OutOfBounds {
            offset: 0,
            required: 1,
            file_size: 0,
        };
        assert!(oob.source().is_none());
    }

    /// From<io::Error> 自动转换为 GgufError::Io。
    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let g: GgufError = io_err.into();
        assert!(matches!(g, GgufError::Io(_)));
        assert!(g.to_string().contains("denied"));
    }
}
