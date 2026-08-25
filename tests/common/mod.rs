//! 测试工具：在内存中构造合法的 GGUF 缓冲，供集成测试使用。
//!
//! 提供低层字节构造原语（write_str / write_scalar_kv / write_array_kv）
//! 与高层缓冲构造（build_gguf_buffer）。

/// 写入 GGUF 字符串（uint64 长度前缀 + 字节）。
pub fn write_str(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

/// 写入一个标量 KV 对（key + type_tag + value_bytes）。
pub fn write_scalar_kv(buf: &mut Vec<u8>, key: &str, type_tag: i32, value_bytes: &[u8]) {
    write_str(buf, key);
    buf.extend_from_slice(&type_tag.to_le_bytes());
    buf.extend_from_slice(value_bytes);
}

/// 写入一个数组 KV 对（key + ARRAY tag + elem_type + count + 元素）。
pub fn write_array_kv(buf: &mut Vec<u8>, key: &str, elem_type: i32, elements: &[Vec<u8>]) {
    write_str(buf, key);
    buf.extend_from_slice(&9i32.to_le_bytes()); // GGUF_TYPE_ARRAY
    buf.extend_from_slice(&elem_type.to_le_bytes());
    buf.extend_from_slice(&(elements.len() as u64).to_le_bytes());
    for e in elements {
        buf.extend_from_slice(e);
    }
}

/// 写入一个张量描述符（name + n_dims + ne[](逻辑序) + dtype + offset）。
///
/// `ne_logical` 为逻辑顺序的维度数组；GGUF 文件中 ne[0..n_dims-1] 按该顺序直接存储。
pub fn write_tensor(buf: &mut Vec<u8>, name: &str, ne_logical: &[i64], dtype: i32, offset: u64) {
    write_str(buf, name);
    let n_dims = ne_logical.len() as u32;
    buf.extend_from_slice(&n_dims.to_le_bytes());
    // 存储序：ne[0..n_dims-1]，即逻辑顺序直接写入
    for d in ne_logical {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// 构造一个完整 GGUF 缓冲：header + kvs + tensors + 假数据体。
///
/// - `kvs`: 已序列化的 KV 字节片段（用 write_scalar_kv / write_array_kv 构造）
/// - `tensors`: (name, ne_logical, dtype, offset)
/// - `data_size`: 假数据体大小（用于测试 data_offset 对齐）
pub fn build_gguf_buffer(
    kv_bytes: &[u8],
    kvs_count: u64,
    tensors: &[(&str, &[i64], i32, u64)],
    data_size: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();
    // header
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
    buf.extend_from_slice(&(kvs_count as i64).to_le_bytes());
    // kvs
    buf.extend_from_slice(kv_bytes);
    // tensors
    for (name, ne, dtype, offset) in tensors {
        write_tensor(&mut buf, name, ne, *dtype, *offset);
    }
    // 假数据体
    let meta_end = buf.len() as u64;
    let align = 32u64;
    let data_start = meta_end.div_ceil(align) * align;
    let total = data_start + data_size;
    buf.resize(total as usize, 0u8);
    buf
}

/// 构造标量值字节（按 gguf_type）。
/// 供 parse_buffer 集成测试使用；其他测试 target 可能不引用，故允许 dead_code。
#[allow(dead_code)]
pub fn scalar_bytes(ty: i32, val: u64) -> Vec<u8> {
    match ty {
        0 => (val as u8).to_le_bytes().to_vec(),
        1 => (val as i8).to_le_bytes().to_vec(),
        2 => (val as u16).to_le_bytes().to_vec(),
        3 => (val as i16).to_le_bytes().to_vec(),
        4 => (val as u32).to_le_bytes().to_vec(),
        5 => (val as i32).to_le_bytes().to_vec(),
        6 => f32::from_bits(val as u32).to_le_bytes().to_vec(),
        7 => (val as i8).to_le_bytes().to_vec(),
        10 => val.to_le_bytes().to_vec(),
        11 => (val as i64).to_le_bytes().to_vec(),
        12 => f64::from_bits(val).to_le_bytes().to_vec(),
        _ => panic!("scalar_bytes: unsupported type {ty}"),
    }
}
