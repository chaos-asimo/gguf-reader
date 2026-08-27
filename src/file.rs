use crate::cursor::Cursor;
use crate::error::{GgufError, GgufResult};
use crate::header::{GgufHeader, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use crate::tensor::TensorInfo;
use crate::types::{GgmlType, GgufArray, GgufType, GgufValue};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// `general.alignment` 键名。
const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";
/// `general.architecture` 键名。
const KEY_GENERAL_ARCHITECTURE: &str = "general.architecture";
/// `general.name` 键名。
const KEY_GENERAL_NAME: &str = "general.name";

/// 张量数据源（用于按需读取张量原始字节）。
#[derive(Clone)]
pub enum DataSource {
    /// 内存缓冲（from_buffer / from_reader / 无 mmap 回退路径）。
    Buffer(Arc<Vec<u8>>),
    /// mmap 视图（from_path + mmap 路径）。
    #[cfg(feature = "mmap")]
    Mmap(Arc<memmap2::Mmap>),
}

impl std::fmt::Debug for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Buffer(v) => f.debug_struct("Buffer").field("len", &v.len()).finish(),
            #[cfg(feature = "mmap")]
            DataSource::Mmap(m) => f.debug_struct("Mmap").field("len", &m.len()).finish(),
        }
    }
}

impl DataSource {
    /// 获取底层字节切片。
    fn as_bytes(&self) -> &[u8] {
        match self {
            DataSource::Buffer(v) => v.as_slice(),
            #[cfg(feature = "mmap")]
            DataSource::Mmap(m) => m.as_ref(),
        }
    }
}

/// 解析后的 GGUF 文件元数据（不含权重数据体）。
#[derive(Clone, Debug)]
pub struct GgufFile {
    pub header: GgufHeader,
    /// KV 元数据，保持文件内出现顺序
    pub kv: Vec<(String, GgufValue)>,
    /// 张量描述符，保持文件内出现顺序
    pub tensors: Vec<TensorInfo>,
    /// 对齐值（来自 general.alignment，缺省 32）
    pub alignment: u32,
    /// 元数据区结束、数据体起始的文件偏移（含对齐填充）
    pub data_offset: u64,
    /// 文件/缓冲总字节数
    pub file_size: u64,
    /// 数据源（用于 tensor_data 按需读取）
    data_source: Option<DataSource>,
}

impl GgufFile {
    /// 从内存缓冲解析 GGUF 元数据。
    ///
    /// 缓冲会被拷贝为 `Arc<Vec<u8>>` 持有，使后续 `tensor_data` 可用。
    pub fn from_buffer(data: &[u8]) -> GgufResult<Self> {
        let file_size = data.len() as u64;
        let mut c = Cursor::new(data);

        // ---- 1. Header ----
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }
        let version = c.u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = c.i64()?;
        if n_tensors < 0 {
            return Err(GgufError::InvalidCount {
                field: "n_tensors",
                value: n_tensors,
            });
        }
        let n_kv = c.i64()?;
        if n_kv < 0 {
            return Err(GgufError::InvalidCount {
                field: "n_kv",
                value: n_kv,
            });
        }
        let header = GgufHeader {
            magic,
            version,
            n_tensors: n_tensors as u64,
            n_kv: n_kv as u64,
        };

        // 合理性上限：防止损坏文件声称天文数字导致长时间解析
        let max_reasonable = file_size; // 每个 KV 至少 8(类型)+1 字节
        if header.n_kv > max_reasonable {
            return Err(GgufError::InvalidCount {
                field: "n_kv",
                value: n_kv,
            });
        }
        if header.n_tensors > max_reasonable {
            return Err(GgufError::InvalidCount {
                field: "n_tensors",
                value: n_tensors,
            });
        }

        // ---- 2. KV Metadata ----
        let mut kv: Vec<(String, GgufValue)> = Vec::with_capacity(header.n_kv as usize);
        for _ in 0..header.n_kv {
            let key = c.string()?;
            let type_i32 = c.i32()?;
            let value = parse_value(&mut c, GgufType::from_i32(type_i32)?)?;
            kv.push((key, value));
        }

        // ---- 3. Tensor Info ----
        let mut tensors: Vec<TensorInfo> =
            Vec::with_capacity(header.n_tensors.min(1_000_000) as usize);
        for _ in 0..header.n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 8 {
                return Err(GgufError::InvalidTensorDim {
                    name: name.clone(),
                    dim: -1,
                });
            }
            // GGUF 文件中 ne[0..n_dims-1] 按逻辑顺序直接存储（与 ggml 实现一致）
            let mut ne = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = c.i64()?;
                if d < 0 {
                    return Err(GgufError::InvalidTensorDim {
                        name: name.clone(),
                        dim: d,
                    });
                }
                ne.push(d as u64);
            }
            let dtype_i32 = c.i32()?;
            let dtype = GgmlType::from_i32(dtype_i32);
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                shape: ne,
                dtype,
                offset,
            });
        }

        // ---- 4. Alignment & data_offset ----
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for (k, v) in &kv {
            if k == KEY_GENERAL_ALIGNMENT {
                if let GgufValue::U32(a) = v {
                    alignment = if *a == 0 { GGUF_DEFAULT_ALIGNMENT } else { *a };
                }
                break;
            }
        }
        let meta_end = c.pos() as u64;
        let data_offset = align_up(meta_end, alignment);

        // 持有数据拷贝，使 tensor_data 可用
        let data_source = Some(DataSource::Buffer(Arc::new(data.to_vec())));

        Ok(GgufFile {
            header,
            kv,
            tensors,
            alignment,
            data_offset,
            file_size,
            data_source,
        })
    }

    /// 从任意 Read+Seek 读取器解析（整体读入内存后调用 from_buffer）。
    pub fn from_reader<R: Read + Seek>(mut reader: R) -> GgufResult<Self> {
        let len = reader.seek(SeekFrom::End(0)).map_err(GgufError::Io)?;
        reader.seek(SeekFrom::Start(0)).map_err(GgufError::Io)?;
        let mut buf = vec![0u8; len as usize];
        reader.read_exact(&mut buf).map_err(GgufError::Io)?;
        Self::from_buffer(&buf)
    }

    /// 读取指定张量的原始数据字节（不反量化）。
    ///
    /// 返回 `Vec<u8>`（owned），数据大小由 `tensor.data_size(dtype, num_elements)` 精确计算。
    /// 若 `data_source` 不存在（如手动构造的 GgufFile），返回 `InferenceError`。
    pub fn tensor_data(&self, tensor: &TensorInfo) -> GgufResult<Vec<u8>> {
        let source = self.data_source.as_ref().ok_or_else(|| {
            GgufError::InferenceError("no data source (file constructed without data)".into())
        })?;
        let bytes = source.as_bytes();
        let data_start = self.data_offset + tensor.offset;
        let data_len = tensor.data_size()? as usize;
        let end = data_start
            .checked_add(data_len as u64)
            .ok_or(GgufError::OutOfBounds {
                offset: data_start,
                required: data_len as u64,
                file_size: self.file_size,
            })?;
        if end > bytes.len() as u64 {
            return Err(GgufError::OutOfBounds {
                offset: data_start,
                required: data_len as u64,
                file_size: self.file_size,
            });
        }
        Ok(bytes[data_start as usize..end as usize].to_vec())
    }

    /// 返回张量数据的精确字节数（基于 dtype block 结构）。
    pub fn tensor_data_size(tensor: &TensorInfo) -> GgufResult<u64> {
        tensor.data_size()
    }

    /// 张量在文件中的物理字节跨度（到 offset 排序后下一张量起始，末张量到文件末尾）。
    ///
    /// **重要**：张量在文件中的物理排列顺序可能不同于 header 列表顺序。
    /// 必须按 offset 升序排序后，取排序后下一个张量的 offset 作为当前张量的 span 终点。
    ///
    /// 某些文件的 dtype 标记与物理量化格式不一致（如 ffn_down 标记 Q8_K 但
    /// 物理占用 210B/256elem = Q6_K 比特率），此时 `data_size()` 偏小。
    /// 反量化前应先比较二者，按物理格式取数。
    pub fn tensor_physical_span(&self, tensor: &TensorInfo) -> GgufResult<u64> {
        if self.tensors.is_empty() {
            return Err(GgufError::InferenceError("no tensors".into()));
        }
        let start = tensor.offset;
        // 按 offset 排序找下一个张量
        let mut sorted: Vec<u64> = self.tensors.iter().map(|t| t.offset).collect();
        sorted.sort();
        let end = match sorted.binary_search(&start) {
            Ok(pos) => {
                if pos + 1 < sorted.len() {
                    sorted[pos + 1]
                } else {
                    let data_end = self.file_size.saturating_sub(self.data_offset);
                    std::cmp::max(data_end, start)
                }
            }
            Err(_) => {
                return Err(GgufError::InferenceError(format!("tensor offset {} not found in sorted list", start)));
            }
        };
        Ok(end.saturating_sub(start))
    }

    /// 读取张量的物理字节（按 [`tensor_physical_span`] 跨度，而非 header `data_size()`）。
    ///
    /// 用于 dtype 标记与物理格式不一致的张量（如 ffn_down）：此时 header size
    /// 偏小，必须按物理跨度取全量字节再反量化。
    pub fn tensor_data_physical(&self, tensor: &TensorInfo) -> GgufResult<Vec<u8>> {
        let source = self.data_source.as_ref().ok_or_else(|| {
            GgufError::InferenceError("no data source (file constructed without data)".into())
        })?;
        let bytes = source.as_bytes();
        let span = self.tensor_physical_span(tensor)?;
        let data_start = self.data_offset + tensor.offset;
        let end = data_start
            .checked_add(span)
            .ok_or(GgufError::OutOfBounds {
                offset: data_start,
                required: span,
                file_size: self.file_size,
            })?;
        if end > bytes.len() as u64 {
            return Err(GgufError::OutOfBounds {
                offset: data_start,
                required: span,
                file_size: self.file_size,
            });
        }
        Ok(bytes[data_start as usize..end as usize].to_vec())
    }

    /// 从文件路径解析。
    ///
    /// 优先使用 mmap（feature `mmap` 开启时），mmap 视图被持有使 `tensor_data` 零拷贝可用；
    /// mmap 失败时回退到整体读入（Buffer 路径）。
    pub fn from_path(path: impl AsRef<Path>) -> GgufResult<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(GgufError::Io)?;
        let file_size = file.metadata().map_err(GgufError::Io)?.len();

        // 先读 header 以确定元数据区大小（避免一次性 mmap 整个大文件后全读）
        // 策略：mmap 整个文件（零拷贝视图），持有 mmap 使 tensor_data 可用
        #[cfg(feature = "mmap")]
        {
            use memmap2::Mmap;
            match unsafe { Mmap::map(&file) } {
                Ok(mmap) => {
                    // 读取 header 计算元数据区长度
                    if mmap.len() < 24 {
                        return Err(GgufError::OutOfBounds {
                            offset: 0,
                            required: 24,
                            file_size: mmap.len() as u64,
                        });
                    }
                    let magic = u32::from_le_bytes(copy_u32(&mmap[0..4]));
                    if magic != GGUF_MAGIC {
                        return Err(GgufError::InvalidMagic(magic));
                    }
                    let version = u32::from_le_bytes(copy_u32(&mmap[4..8]));
                    if version != GGUF_VERSION {
                        return Err(GgufError::UnsupportedVersion(version));
                    }
                    let n_tensors = i64::from_le_bytes(copy_u64(&mmap[8..16]));
                    let n_kv = i64::from_le_bytes(copy_u64(&mmap[16..24]));
                    if n_tensors < 0 || n_kv < 0 {
                        return Err(GgufError::InvalidCount {
                            field: "counts",
                            value: n_kv.min(n_tensors),
                        });
                    }
                    // 对 mmap 视图解析元数据
                    let mut parsed = Self::from_buffer(&mmap[..])?;
                    // 替换 data_source 为 mmap 持有（零拷贝 tensor_data）
                    parsed.data_source = Some(DataSource::Mmap(Arc::new(mmap)));
                    Ok(parsed)
                }
                Err(_e) => {
                    // mmap 失败，回退到整体读取
                    Self::read_from_file(&file, file_size)
                }
            }
        }

        #[cfg(not(feature = "mmap"))]
        {
            let _ = &file;
            Self::read_from_file(&file, file_size)
        }
    }

    /// 整体读入文件的回退路径。
    fn read_from_file(file: &File, file_size: u64) -> GgufResult<Self> {
        let mut buf = vec![0u8; file_size as usize];
        {
            let mut f = file.try_clone().map_err(GgufError::Io)?;
            f.read_exact(&mut buf).map_err(GgufError::Io)?;
        }
        Self::from_buffer(&buf)
    }

    // ---------- 便捷方法 ----------

    /// 按键查找 KV 值（首次出现）。
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// 按名查找张量（首次出现）。
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 架构名（general.architecture）。
    pub fn architecture(&self) -> Option<&str> {
        self.get(KEY_GENERAL_ARCHITECTURE).and_then(|v| v.as_str())
    }

    /// 模型名（general.name）。
    pub fn model_name(&self) -> Option<&str> {
        self.get(KEY_GENERAL_NAME).and_then(|v| v.as_str())
    }

    /// KV 转 HashMap（便于 JSON 序列化与快速查询；重复键后者覆盖前者）。
    pub fn kv_map(&self) -> HashMap<&str, &GgufValue> {
        let mut m = HashMap::with_capacity(self.kv.len());
        for (k, v) in &self.kv {
            m.insert(k.as_str(), v);
        }
        m
    }
}

/// 对齐上取整：返回不小于 `value` 且为 `align` 倍数的值。align 为 0 时原样返回。
fn align_up(value: u64, align: u32) -> u64 {
    if align == 0 {
        return value;
    }
    let a = u64::from(align);
    value.div_ceil(a) * a
}

/// 从 4 字节切片拷贝为 [u8; 4]（调用前须确保切片长度恰好为 4）。
#[cfg_attr(not(feature = "mmap"), allow(dead_code))]
fn copy_u32(s: &[u8]) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(s);
    out
}

/// 从 8 字节切片拷贝为 [u8; 8]（调用前须确保切片长度恰好为 8）。
#[cfg_attr(not(feature = "mmap"), allow(dead_code))]
fn copy_u64(s: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(s);
    out
}

/// 解析单个 KV 值（依据类型标签）。
fn parse_value(c: &mut Cursor, ty: GgufType) -> GgufResult<GgufValue> {
    match ty {
        GgufType::Uint8 => Ok(GgufValue::U8(c.u8()?)),
        GgufType::Int8 => Ok(GgufValue::I8(c.i8()?)),
        GgufType::Uint16 => Ok(GgufValue::U16(c.u16()?)),
        GgufType::Int16 => Ok(GgufValue::I16(c.i16()?)),
        GgufType::Uint32 => Ok(GgufValue::U32(c.u32()?)),
        GgufType::Int32 => Ok(GgufValue::I32(c.i32()?)),
        GgufType::F32 => Ok(GgufValue::F32(c.f32()?)),
        GgufType::Bool => Ok(GgufValue::Bool(c.bool()?)),
        GgufType::String => Ok(GgufValue::String(c.string()?)),
        GgufType::Uint64 => Ok(GgufValue::U64(c.u64()?)),
        GgufType::Int64 => Ok(GgufValue::I64(c.i64()?)),
        GgufType::F64 => Ok(GgufValue::F64(c.f64()?)),
        GgufType::Array => parse_array(c),
    }
}

/// 解析数组值。
fn parse_array(c: &mut Cursor) -> GgufResult<GgufValue> {
    let elem_i32 = c.i32()?;
    let elem_type = GgufType::from_i32(elem_i32)?;
    if elem_type == GgufType::Array {
        return Err(GgufError::InvalidArrayElemType(elem_i32));
    }
    let count = c.u64()?;

    // 防 OOM 预检：count * min_element_size 不能超过剩余字节
    let min_size = elem_type.min_element_size();
    let required = count.saturating_mul(min_size);
    if required > c.remaining() as u64 {
        return Err(GgufError::OutOfBounds {
            offset: c.pos() as u64,
            required,
            file_size: c.total_len(),
        });
    }

    let mut data = Vec::with_capacity(count.min(1_000_000) as usize);
    for _ in 0..count {
        data.push(parse_value(c, elem_type)?);
    }
    Ok(GgufValue::Array(GgufArray { elem_type, data }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小合法 GGUF 缓冲（用于单元测试）。
    pub(crate) fn build_test_gguf(
        kvs: &[(Vec<u8>, Vec<u8>)], // (key_bytes, value_bytes_incl_type)
        tensors: &[(Vec<u8>, Vec<i64>, i32, u64)], // (name, ne, dtype, offset)
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as i64).to_le_bytes());
        // kvs
        for (k, v) in kvs {
            write_str(&mut buf, k);
            buf.extend_from_slice(v);
        }
        // tensors
        for (name, ne, dtype, offset) in tensors {
            write_str(&mut buf, name);
            buf.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            for d in ne {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        buf
    }

    pub(crate) fn write_str(buf: &mut Vec<u8>, s: &[u8]) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s);
    }

    /// 构造一个 u32 类型的 KV 值字节（含类型标签）。
    pub(crate) fn kv_u32(key: &str, val: u32) -> (Vec<u8>, Vec<u8>) {
        let mut v = Vec::new();
        v.extend_from_slice(&4i32.to_le_bytes()); // GGUF_TYPE_UINT32
        v.extend_from_slice(&val.to_le_bytes());
        (key.as_bytes().to_vec(), v)
    }

    /// 构造一个 string 类型的 KV 值字节（含类型标签）。
    pub(crate) fn kv_str(key: &str, val: &str) -> (Vec<u8>, Vec<u8>) {
        let mut v = Vec::new();
        v.extend_from_slice(&8i32.to_le_bytes()); // GGUF_TYPE_STRING
        v.extend_from_slice(&(val.len() as u64).to_le_bytes());
        v.extend_from_slice(val.as_bytes());
        (key.as_bytes().to_vec(), v)
    }

    #[test]
    fn test_parse_minimal() {
        let kvs = vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
        ];
        let tensors = vec![
            ("tok".as_bytes().to_vec(), vec![128, 4096], 30, 0), // BF16
            ("out".as_bytes().to_vec(), vec![4096], 0, 1024),    // F32
        ];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.header.version, 3);
        assert_eq!(f.header.n_kv, 2);
        assert_eq!(f.header.n_tensors, 2);
        assert_eq!(f.architecture(), Some("llama"));
        assert_eq!(f.get("llama.block_count").unwrap(), &GgufValue::U32(32));
        assert_eq!(f.tensors[0].name, "tok");
        assert_eq!(f.tensors[0].shape, vec![128, 4096]);
        assert_eq!(f.tensors[0].dtype, GgmlType::BF16);
        assert_eq!(f.tensors[0].offset, 0);
        assert_eq!(f.tensors[1].name, "out");
        assert_eq!(f.tensors[1].shape, vec![4096]);
        assert_eq!(f.tensors[1].dtype, GgmlType::F32);
        assert_eq!(f.alignment, 32);
        assert!(f.data_offset >= buf.len() as u64);
        assert_eq!(f.file_size, buf.len() as u64);
    }

    #[test]
    fn test_alignment_from_kv() {
        let kvs = vec![kv_u32("general.alignment", 16)];
        let tensors: Vec<_> = vec![];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.alignment, 16);
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"XXXX");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        assert!(matches!(
            GgufFile::from_buffer(&buf),
            Err(GgufError::InvalidMagic(_))
        ));
    }

    #[test]
    fn test_invalid_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        assert!(matches!(
            GgufFile::from_buffer(&buf),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn test_empty_buffer() {
        assert!(matches!(
            GgufFile::from_buffer(&[]),
            Err(GgufError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_truncated_header() {
        let buf = [0x47, 0x47]; // 仅 2 字节
        assert!(GgufFile::from_buffer(&buf).is_err());
    }

    #[test]
    fn test_negative_n_kv() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&(-1i64).to_le_bytes());
        assert!(matches!(
            GgufFile::from_buffer(&buf),
            Err(GgufError::InvalidCount { .. })
        ));
    }

    /// 张量维度数 > 8 应返回 InvalidTensorDim（dim=-1 标记）。
    #[test]
    fn test_tensor_dims_too_many() {
        // 9 个维度，每个 1
        let dims = vec![1i64; 9];
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("t".as_bytes().to_vec(), dims, 0i32, 0u64)];
        let buf = build_test_gguf(&kvs, &tensors);
        let err = GgufFile::from_buffer(&buf).unwrap_err();
        assert!(matches!(
            err,
            GgufError::InvalidTensorDim { name: _, dim: -1 }
        ));
    }

    /// 0 维张量（标量）合法：shape 为空，num_elements=1。
    #[test]
    fn test_tensor_zero_dims() {
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("s".as_bytes().to_vec(), Vec::<i64>::new(), 0i32, 0u64)];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.tensors[0].name, "s");
        assert!(f.tensors[0].shape.is_empty());
        assert_eq!(f.tensors[0].num_elements(), 1);
    }

    /// 5 维与 8 维张量合法（≤8 边界）。
    #[test]
    fn test_tensor_high_dims() {
        let dims5 = vec![2i64, 3, 4, 5, 6];
        let dims8 = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        // 手工构造两个张量（build_test_gguf 也支持，但这里直接控制字节）
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&2i64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0i64.to_le_bytes()); // n_kv
        for (name, dims) in [("a", &dims5), ("b", &dims8)] {
            write_str(&mut buf, name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&0i32.to_le_bytes()); // dtype F32
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.tensors[0].shape, vec![2u64, 3, 4, 5, 6]);
        assert_eq!(f.tensors[1].shape, vec![1u64, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// general.alignment = 0 应回退到默认 32。
    #[test]
    fn test_alignment_zero_falls_back() {
        let kvs = vec![kv_u32("general.alignment", 0)];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> = vec![];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.alignment, GGUF_DEFAULT_ALIGNMENT);
    }

    /// n_kv 声称超过文件大小的合理性上限应报 InvalidCount。
    #[test]
    fn test_n_kv_exceeds_file_size() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1000000i64.to_le_bytes()); // n_kv 远超文件大小
        let err = GgufFile::from_buffer(&buf).unwrap_err();
        assert!(matches!(err, GgufError::InvalidCount { .. }));
    }

    /// n_tensors 负值应报 InvalidCount。
    #[test]
    fn test_negative_n_tensors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(-5i64).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        assert!(matches!(
            GgufFile::from_buffer(&buf),
            Err(GgufError::InvalidCount {
                field: "n_tensors",
                ..
            })
        ));
    }

    /// from_reader 与 from_buffer 等价（用 Cursor 模拟 Read+Seek）。
    #[test]
    fn test_from_reader() {
        use std::io::Cursor as IoCursor;
        let kvs = vec![kv_str("general.architecture", "bert")];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("t".as_bytes().to_vec(), vec![8], 0, 0)];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_reader(IoCursor::new(buf.clone())).unwrap();
        assert_eq!(f.architecture(), Some("bert"));
        assert_eq!(f.file_size, buf.len() as u64);
        assert_eq!(f.header.n_tensors, 1);
    }

    /// from_reader 对损坏数据同样返回错误。
    #[test]
    fn test_from_reader_corrupt() {
        use std::io::Cursor as IoCursor;
        let bad = vec![0x00u8; 24]; // magic 不对
        let err = GgufFile::from_reader(IoCursor::new(bad)).unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic(_)));
    }

    /// from_path 成功解析真实临时文件，并与 from_buffer 一致。
    #[test]
    fn test_from_path_roundtrip() {
        let kvs = vec![
            kv_str("general.architecture", "llama"),
            kv_str("general.name", "PathModel"),
        ];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("w".as_bytes().to_vec(), vec![16, 16], 0, 0)];
        let buf = build_test_gguf(&kvs, &tensors);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("path_test.gguf");
        std::fs::write(&path, &buf).unwrap();

        let f = GgufFile::from_path(&path).unwrap();
        assert_eq!(f.architecture(), Some("llama"));
        assert_eq!(f.model_name(), Some("PathModel"));
        assert_eq!(f.file_size, buf.len() as u64);
        assert_eq!(f.tensors.len(), 1);
    }

    /// from_path 对不存在的文件返回 Io 错误。
    #[test]
    fn test_from_path_missing() {
        let err = GgufFile::from_path("/nonexistent/definitely_missing.gguf").unwrap_err();
        assert!(matches!(err, GgufError::Io(_)));
    }

    /// data_offset 对齐上取整的正确性（手工计算）。
    #[test]
    fn test_data_offset_alignment_math() {
        // 构造已知 meta_end 的缓冲，验证 data_offset 是 alignment 的倍数
        let kvs = vec![kv_u32("general.alignment", 8)];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("x".as_bytes().to_vec(), vec![1], 0, 0)];
        let buf = build_test_gguf(&kvs, &tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        assert_eq!(f.alignment, 8);
        assert_eq!(f.data_offset % 8, 0);
        assert!(f.data_offset >= f.file_size);
    }

    /// 构造含真实张量数据体的 GGUF 缓冲：metadata 后 padding 到 data_offset，再写 data。
    fn build_gguf_with_data(
        kvs: &[(Vec<u8>, Vec<u8>)],
        tensors: &[(Vec<u8>, Vec<i64>, i32, u64)],
        data: &[u8],
    ) -> Vec<u8> {
        let mut buf = build_test_gguf(kvs, tensors);
        let f = GgufFile::from_buffer(&buf).unwrap();
        let pad_len = f.data_offset as usize - buf.len();
        buf.extend(std::iter::repeat_n(0u8, pad_len));
        buf.extend_from_slice(data);
        buf
    }

    /// tensor_data：buffer 路径成功读取（F32，1 个 4 字节元素）。
    #[test]
    fn test_tensor_data_buffer() {
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("w".as_bytes().to_vec(), vec![4], 0, 0)]; // F32, 4 elems
        let data: Vec<u8> = (0..4i32).flat_map(|i| i.to_le_bytes()).collect();
        let buf = build_gguf_with_data(&kvs, &tensors, &data);
        let f = GgufFile::from_buffer(&buf).unwrap();
        let t = f.find_tensor("w").unwrap();
        let got = f.tensor_data(t).unwrap();
        assert_eq!(got, data);
        assert_eq!(GgufFile::tensor_data_size(t).unwrap(), 16);
    }

    /// tensor_data：mmap 路径（from_path）读取与 buffer 路径一致。
    #[test]
    fn test_tensor_data_mmap_path() {
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("w".as_bytes().to_vec(), vec![2, 2], 0, 0)]; // F32, 4 elems
        let data: Vec<u8> = [1f32, -2.5, 3.25, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let buf = build_gguf_with_data(&kvs, &tensors, &data);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        let f = GgufFile::from_path(&path).unwrap();
        let t = f.find_tensor("w").unwrap();
        assert_eq!(f.tensor_data(t).unwrap(), data);
    }

    /// tensor_data：数据体截断应报 OutOfBounds。
    #[test]
    fn test_tensor_data_truncated() {
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let tensors: Vec<(Vec<u8>, Vec<i64>, i32, u64)> =
            vec![("w".as_bytes().to_vec(), vec![4], 0, 0)]; // 需要 16 字节
        let data = vec![0u8; 8]; // 仅 8 字节
        let buf = build_gguf_with_data(&kvs, &tensors, &data);
        let f = GgufFile::from_buffer(&buf).unwrap();
        let t = f.find_tensor("w").unwrap();
        assert!(matches!(
            f.tensor_data(t),
            Err(GgufError::OutOfBounds { .. })
        ));
    }
}
