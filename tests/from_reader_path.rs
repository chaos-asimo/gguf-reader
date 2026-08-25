//! 集成测试：验证 from_reader / from_path 两种入口与 from_buffer 行为一致。
//!
//! from_reader 接受任意 Read+Seek（此处用 std::io::Cursor）；
//! from_path 从真实文件路径解析（mmap 优先，失败回退整体读取）。

mod common;

use common::*;
use gguf::GgufFile;
use std::io::Cursor as IoCursor;

/// 构造一个含 KV + 张量的合法缓冲（复用 common 工具）。
fn sample_buffer() -> Vec<u8> {
    let mut kv = Vec::new();
    write_scalar_kv(&mut kv, "general.architecture", 8, &str_bytes("llama"));
    write_scalar_kv(&mut kv, "general.name", 8, &str_bytes("ReaderModel"));
    write_scalar_kv(&mut kv, "llama.block_count", 4, &16u32.to_le_bytes());
    let tensors: Vec<(&str, &[i64], i32, u64)> =
        vec![("tok", &[128, 512], 30, 0), ("out", &[512], 0, 262144)];
    build_gguf_buffer(&kv, 3, &tensors, 0)
}

fn str_bytes(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut v = (b.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(b);
    v
}

/// from_reader 解析成功，字段与 from_buffer 一致。
#[test]
fn test_from_reader_success() {
    let buf = sample_buffer();
    let via_reader = GgufFile::from_reader(IoCursor::new(buf.clone())).unwrap();
    let via_buffer = GgufFile::from_buffer(&buf).unwrap();

    assert_eq!(via_reader.architecture(), Some("llama"));
    assert_eq!(via_reader.model_name(), Some("ReaderModel"));
    assert_eq!(via_reader.header.n_kv, 3);
    assert_eq!(via_reader.header.n_tensors, 2);
    assert_eq!(via_reader.file_size, via_buffer.file_size);
    assert_eq!(via_reader.data_offset, via_buffer.data_offset);
    assert_eq!(via_reader.tensors.len(), 2);
    assert_eq!(via_reader.tensors[0].shape, vec![128, 512]);
}

/// from_reader 对空缓冲返回错误（与 from_buffer 一致）。
#[test]
fn test_from_reader_empty() {
    let res = GgufFile::from_reader(IoCursor::new(Vec::<u8>::new()));
    assert!(res.is_err());
}

/// from_reader 对损坏 magic 返回错误。
#[test]
fn test_from_reader_bad_magic() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"NOPE");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    let res = GgufFile::from_reader(IoCursor::new(buf));
    assert!(res.is_err());
}

/// from_path 解析真实临时文件，结果与 from_buffer 一致。
#[test]
fn test_from_path_success() {
    let buf = sample_buffer();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reader_path.gguf");
    std::fs::write(&path, &buf).unwrap();

    let via_path = GgufFile::from_path(&path).unwrap();
    let via_buffer = GgufFile::from_buffer(&buf).unwrap();

    assert_eq!(via_path.architecture(), Some("llama"));
    assert_eq!(via_path.model_name(), Some("ReaderModel"));
    assert_eq!(via_path.file_size, via_buffer.file_size);
    assert_eq!(via_path.data_offset, via_buffer.data_offset);
    assert_eq!(via_path.alignment, via_buffer.alignment);
    assert_eq!(via_path.tensors.len(), via_buffer.tensors.len());
    // 张量逐项一致
    for (a, b) in via_path.tensors.iter().zip(via_buffer.tensors.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.offset, b.offset);
    }
}

/// from_path 对不存在文件返回 Io 错误。
#[test]
fn test_from_path_missing() {
    let res = GgufFile::from_path("/definitely/does/not/exist.gguf");
    assert!(res.is_err());
}

/// from_path 对非法 GGUF 内容（如空文件）返回错误。
#[test]
fn test_from_path_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    std::fs::write(&path, b"").unwrap();
    let res = GgufFile::from_path(&path);
    assert!(res.is_err());
}
