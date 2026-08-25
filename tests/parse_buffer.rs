//! 集成测试：基于内存缓冲的 GGUF 解析（覆盖全部 KV 类型、张量、对齐、错误路径）。

mod common;

use common::*;
use gguf::*;

/// 构造包含全部 13 种 KV 类型 + 数组的测试缓冲，验证解析结果。
#[test]
fn test_all_kv_types() {
    let mut kv = Vec::new();

    // UINT8 = 0
    write_scalar_kv(&mut kv, "k.u8", 0, &scalar_bytes(0, 7));
    // INT8 = 1 (负值)
    write_scalar_kv(&mut kv, "k.i8", 1, &(-5i8).to_le_bytes());
    // UINT16 = 2
    write_scalar_kv(&mut kv, "k.u16", 2, &scalar_bytes(2, 0xBEEF));
    // INT16 = 3 (负值)
    write_scalar_kv(&mut kv, "k.i16", 3, &(-1234i16).to_le_bytes());
    // UINT32 = 4
    write_scalar_kv(&mut kv, "k.u32", 4, &scalar_bytes(4, 0xDEAD_BEEF));
    // INT32 = 5 (负值)
    write_scalar_kv(&mut kv, "k.i32", 5, &(-99999i32).to_le_bytes());
    // FLOAT32 = 6
    write_scalar_kv(&mut kv, "k.f32", 6, &3.5f32.to_le_bytes());
    // BOOL = 7
    write_scalar_kv(&mut kv, "k.bool", 7, &1i8.to_le_bytes());
    // STRING = 8
    write_scalar_kv(&mut kv, "k.str", 8, &string_value_bytes("hello world"));
    // ARRAY of F32 = 9 (elem 6)
    let mut arr = Vec::new();
    for i in 0..4 {
        arr.extend_from_slice(&((i as f32) * 1.5).to_le_bytes());
    }
    write_array_kv(&mut kv, "k.f32arr", 6, &chunks_of(&arr, 4));
    // UINT64 = 10
    write_scalar_kv(
        &mut kv,
        "k.u64",
        10,
        &0x1122_3344_5566_7788u64.to_le_bytes(),
    );
    // INT64 = 11 (负值)
    write_scalar_kv(&mut kv, "k.i64", 11, &(-12345678901234i64).to_le_bytes());
    // FLOAT64 = 12
    write_scalar_kv(&mut kv, "k.f64", 12, &2.25f64.to_le_bytes());

    // 13 个 KV
    let tensors: Vec<(&str, &[i64], i32, u64)> =
        vec![("t1", &[100, 200], 0, 0), ("t2", &[50], 30, 8000)];
    let buf = build_gguf_buffer(&kv, 13, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).expect("parse should succeed");

    assert_eq!(f.header.n_kv, 13);
    assert_eq!(f.header.n_tensors, 2);
    assert_eq!(f.kv.len(), 13);

    // 逐项断言
    assert_eq!(f.get("k.u8").unwrap(), &GgufValue::U8(7));
    assert_eq!(f.get("k.i8").unwrap(), &GgufValue::I8(-5));
    assert_eq!(f.get("k.u16").unwrap(), &GgufValue::U16(0xBEEF));
    assert_eq!(f.get("k.i16").unwrap(), &GgufValue::I16(-1234));
    assert_eq!(f.get("k.u32").unwrap(), &GgufValue::U32(0xDEAD_BEEF));
    assert_eq!(f.get("k.i32").unwrap(), &GgufValue::I32(-99999));
    assert!((f.get("k.f32").unwrap().as_f64().unwrap() - 3.5).abs() < 1e-6);
    assert_eq!(f.get("k.bool").unwrap(), &GgufValue::Bool(true));
    assert_eq!(
        f.get("k.str").unwrap(),
        &GgufValue::String("hello world".into())
    );

    let arr_val = f.get("k.f32arr").unwrap().as_array().unwrap();
    assert_eq!(arr_val.elem_type, GgufType::F32);
    assert_eq!(arr_val.data.len(), 4);
    assert!((arr_val.data[2].as_f64().unwrap() - 3.0).abs() < 1e-6);

    assert_eq!(
        f.get("k.u64").unwrap(),
        &GgufValue::U64(0x1122_3344_5566_7788)
    );
    assert_eq!(f.get("k.i64").unwrap(), &GgufValue::I64(-12345678901234));
    assert!((f.get("k.f64").unwrap().as_f64().unwrap() - 2.25).abs() < 1e-12);

    // 张量
    assert_eq!(f.tensors[0].name, "t1");
    assert_eq!(f.tensors[0].shape, vec![100, 200]);
    assert_eq!(f.tensors[0].dtype, GgmlType::F32);
    assert_eq!(f.tensors[0].offset, 0);
    assert_eq!(f.tensors[0].num_elements(), 20000);

    assert_eq!(f.tensors[1].name, "t2");
    assert_eq!(f.tensors[1].shape, vec![50]);
    assert_eq!(f.tensors[1].dtype, GgmlType::BF16);
    assert_eq!(f.tensors[1].offset, 8000);
}

/// 字符串值字节（uint64 长度 + UTF-8）。
fn string_value_bytes(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut v = (b.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(b);
    v
}

/// 把字节切片按 chunk 大小分组（用于数组元素）。
fn chunks_of(bytes: &[u8], chunk: usize) -> Vec<Vec<u8>> {
    bytes.chunks(chunk).map(|c| c.to_vec()).collect()
}

#[test]
fn test_alignment_default_and_custom() {
    // 缺省对齐 32
    let kv = Vec::new();
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![("a", &[10], 0, 0)];
    let buf = build_gguf_buffer(&kv, 0, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();
    assert_eq!(f.alignment, 32);

    // 自定义对齐 16
    let mut kv = Vec::new();
    write_scalar_kv(&mut kv, "general.alignment", 4, &16u32.to_le_bytes());
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();
    assert_eq!(f.alignment, 16);
}

#[test]
fn test_data_offset_alignment() {
    let mut kv = Vec::new();
    write_scalar_kv(&mut kv, "general.alignment", 4, &8u32.to_le_bytes());
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![("x", &[4], 0, 0)];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 100);
    let f = GgufFile::from_buffer(&buf).unwrap();
    // data_offset 应为 8 的倍数
    assert_eq!(f.data_offset % 8, 0);
    assert!(f.data_offset <= f.file_size);
}

#[test]
fn test_unknown_dtype_degrades() {
    let kv = Vec::new();
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![("z", &[2, 2], 99, 0)];
    let buf = build_gguf_buffer(&kv, 0, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();
    assert_eq!(f.tensors[0].dtype, GgmlType::Unknown(99));
    assert_eq!(f.tensors[0].dtype.to_string(), "UnknownType(99)");
}

#[test]
fn test_invalid_magic() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"NOTG");
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
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&42u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::UnsupportedVersion(42))
    ));
}

#[test]
fn test_empty_and_short() {
    assert!(GgufFile::from_buffer(&[]).is_err());
    assert!(GgufFile::from_buffer(&[0x47]).is_err());
    // 仅 header 24 字节，n_kv=0 n_tensors=0 应成功
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    let f = GgufFile::from_buffer(&buf).expect("header-only should parse");
    assert_eq!(f.kv.len(), 0);
    assert_eq!(f.tensors.len(), 0);
}

#[test]
fn test_truncated_kv() {
    // 声称 1 个 KV，但后面没有任何字节
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&1i64.to_le_bytes()); // n_kv = 1
                                                // 缺失 KV 内容 → OutOfBounds
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::OutOfBounds { .. })
    ));
}

#[test]
fn test_corrupt_array_count() {
    // 数组声称 1000 个元素，但只有 4 字节 → OutOfBounds
    let mut kv = Vec::new();
    write_str(&mut kv, "big");
    kv.extend_from_slice(&9i32.to_le_bytes()); // ARRAY
    kv.extend_from_slice(&0i32.to_le_bytes()); // elem UINT8
    kv.extend_from_slice(&1000u64.to_le_bytes()); // count
    kv.extend_from_slice(&[1, 2, 3, 4]); // 仅 4 字节
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::OutOfBounds { .. })
    ));
}

#[test]
fn test_nested_array_rejected() {
    // 数组元素类型为 ARRAY(9) → InvalidArrayElemType
    let mut kv = Vec::new();
    write_str(&mut kv, "nested");
    kv.extend_from_slice(&9i32.to_le_bytes()); // ARRAY
    kv.extend_from_slice(&9i32.to_le_bytes()); // elem ARRAY (非法)
    kv.extend_from_slice(&0u64.to_le_bytes()); // count 0
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::InvalidArrayElemType(9))
    ));
}

#[test]
fn test_invalid_utf8_string() {
    // 字符串含非法 UTF-8
    let mut kv = Vec::new();
    write_str(&mut kv, "bad");
    kv.extend_from_slice(&8i32.to_le_bytes()); // STRING
    kv.extend_from_slice(&3u64.to_le_bytes()); // len 3
    kv.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // 非法 UTF-8
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::InvalidStringLength(_))
    ));
}

#[test]
fn test_negative_tensor_dim() {
    let kv = Vec::new();
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![("neg", &[-5, 3], 0, 0)];
    let buf = build_gguf_buffer(&kv, 0, &tensors, 0);
    assert!(matches!(
        GgufFile::from_buffer(&buf),
        Err(GgufError::InvalidTensorDim { .. })
    ));
}

#[test]
fn test_kv_order_preserved() {
    let mut kv = Vec::new();
    write_scalar_kv(&mut kv, "first", 4, &1u32.to_le_bytes());
    write_scalar_kv(&mut kv, "second", 4, &2u32.to_le_bytes());
    write_scalar_kv(&mut kv, "third", 4, &3u32.to_le_bytes());
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 3, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();
    assert_eq!(f.kv[0].0, "first");
    assert_eq!(f.kv[1].0, "second");
    assert_eq!(f.kv[2].0, "third");
}

#[test]
fn test_get_and_find_tensor() {
    let mut kv = Vec::new();
    write_scalar_kv(
        &mut kv,
        "general.architecture",
        8,
        &string_value_bytes("llama"),
    );
    write_scalar_kv(&mut kv, "general.name", 8, &string_value_bytes("TestModel"));
    let tensors: Vec<(&str, &[i64], i32, u64)> =
        vec![("tok", &[10], 0, 0), ("attn", &[4, 4], 1, 40)];
    let buf = build_gguf_buffer(&kv, 2, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();

    assert_eq!(f.architecture(), Some("llama"));
    assert_eq!(f.model_name(), Some("TestModel"));
    assert_eq!(f.get("nonexistent"), None);
    assert!(f.find_tensor("attn").is_some());
    assert_eq!(f.find_tensor("attn").unwrap().offset, 40);
    assert!(f.find_tensor("nope").is_none());

    let map = f.kv_map();
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("general.architecture"));
}

#[test]
fn test_string_array() {
    // 数组 of STRING，验证 token 列表解析
    let elems: Vec<Vec<u8>> = vec![
        string_value_bytes("<s>"),
        string_value_bytes("</s>"),
        string_value_bytes("hello"),
    ];
    let mut kv = Vec::new();
    write_array_kv(&mut kv, "tokenizer.ggml.tokens", 8, &elems);
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();

    let arr = f.get("tokenizer.ggml.tokens").unwrap().as_array().unwrap();
    assert_eq!(arr.elem_type, GgufType::String);
    assert_eq!(arr.data.len(), 3);
    assert_eq!(arr.data[0], GgufValue::String("<s>".into()));
    assert_eq!(arr.data[2], GgufValue::String("hello".into()));
}

#[test]
fn test_tensor_est_data_size() {
    let kv = Vec::new();
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![
        ("f32", &[10, 20], 0, 0),
        ("bf16", &[100], 30, 0),
        ("q4k", &[100, 100], 14, 0),
    ];
    let buf = build_gguf_buffer(&kv, 0, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();
    assert_eq!(f.tensors[0].est_data_size(), Some(200 * 4));
    assert_eq!(f.tensors[1].est_data_size(), Some(100 * 2));
    assert_eq!(f.tensors[2].est_data_size(), None); // 量化类型不估算
}
