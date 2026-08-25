/// GGUF 文件头。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgufHeader {
    /// 魔数，应等于 [`GGUF_MAGIC`]
    pub magic: u32,
    /// 格式版本号，当前为 [`GGUF_VERSION`]
    pub version: u32,
    /// 张量数量
    pub n_tensors: u64,
    /// 键值对数量
    pub n_kv: u64,
}

/// GGUF 魔数（"GGUF" 的 ASCII 小端 u32 表示）。
pub const GGUF_MAGIC: u32 = 0x46554747;

/// 当前支持的 GGUF 版本号。
pub const GGUF_VERSION: u32 = 3;

/// 默认张量数据对齐（字节）。
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
