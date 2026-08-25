//! 集成测试：通过公共 API 验证 value_to_json 的 JSON 序列化。
//!
//! 覆盖全部 13 种标量变体 + 数组（截断/不截断/None 限制）。
//! 单元层已做等价断言，此处从「解析 → 序列化」端到端角度再验证一次，
//! 确保 value_to_json 与 GgufValue 解析路径协同正确。

mod common;

use common::*;
use gguf::{value_to_json, GgufArray, GgufFile, GgufType, GgufValue};
use serde_json::json;

/// 用 value_to_json 直接序列化构造出的 GgufValue，核对 JSON 结构。
#[test]
fn test_json_scalars_direct() {
    assert_eq!(value_to_json(&GgufValue::U8(200), None), json!(200));
    assert_eq!(value_to_json(&GgufValue::I8(-100), None), json!(-100));
    assert_eq!(value_to_json(&GgufValue::U16(60000), None), json!(60000));
    assert_eq!(value_to_json(&GgufValue::I16(-30000), None), json!(-30000));
    assert_eq!(
        value_to_json(&GgufValue::U32(4_000_000_000u32), None),
        json!(4_000_000_000u32)
    );
    assert_eq!(
        value_to_json(&GgufValue::I32(-2_000_000_000i32), None),
        json!(-2_000_000_000i32)
    );
    assert_eq!(
        value_to_json(&GgufValue::U64(18_446_744_073_709_551_615u64), None),
        json!(18_446_744_073_709_551_615u64)
    );
    assert_eq!(
        value_to_json(&GgufValue::I64(-9_007_199_254_740_993i64), None),
        json!(-9_007_199_254_740_993i64)
    );
    assert_eq!(value_to_json(&GgufValue::Bool(true), None), json!(true));
    assert_eq!(
        value_to_json(&GgufValue::String("中文测试".to_string()), None),
        json!("中文测试")
    );

    // 浮点：JSON number 比较
    let f32 = value_to_json(&GgufValue::F32(-1.25), None);
    assert_eq!(f32.as_f64(), Some(-1.25));
    let f64 = value_to_json(&GgufValue::F64(0.1), None);
    assert!((f64.as_f64().unwrap() - 0.1).abs() < 1e-15);
}

/// 数组序列化：element_type / count / value 三字段齐全。
#[test]
fn test_json_array_structure() {
    let arr = GgufValue::Array(GgufArray {
        elem_type: GgufType::String,
        data: vec![
            GgufValue::String("<s>".into()),
            GgufValue::String("hi".into()),
            GgufValue::String("</s>".into()),
        ],
    });
    let v = value_to_json(&arr, None);
    assert_eq!(v["element_type"], "string");
    assert_eq!(v["count"], 3);
    assert_eq!(v["value"], json!(["<s>", "hi", "</s>"]));
    assert!(v.get("truncated").is_none());
    assert!(v.get("total").is_none());
}

/// 超阈值数组：truncated=true，total=原长度，value 仅前 N 项。
#[test]
fn test_json_array_truncation_fields() {
    let data: Vec<GgufValue> = (0..2000).map(|i| GgufValue::I64(i as i64)).collect();
    let arr = GgufValue::Array(GgufArray {
        elem_type: GgufType::Int64,
        data,
    });
    let v = value_to_json(&arr, Some(500));
    assert_eq!(v["count"], 2000);
    assert_eq!(v["truncated"], true);
    assert_eq!(v["total"], 2000);
    let shown = v["value"].as_array().unwrap();
    assert_eq!(shown.len(), 500);
    assert_eq!(shown[0], json!(0));
    assert_eq!(shown[499], json!(499));
}

/// 解析一个含多种类型 KV 的真实缓冲，再整体序列化，验证端到端一致。
#[test]
fn test_roundtrip_parse_then_json() {
    let mut kv = Vec::new();
    write_scalar_kv(&mut kv, "k.u32", 4, &7u32.to_le_bytes());
    write_scalar_kv(&mut kv, "k.str", 8, &string_value_bytes("hello"));
    write_scalar_kv(&mut kv, "k.bool", 7, &1i8.to_le_bytes());
    write_scalar_kv(&mut kv, "k.f32", 6, &3.5f32.to_le_bytes());
    // 数组 of f32（4 个元素）
    let mut arr_bytes = Vec::new();
    for i in 0..4 {
        arr_bytes.extend_from_slice(&((i as f32) * 0.5).to_le_bytes());
    }
    write_array_kv(&mut kv, "k.f32arr", 6, &chunks(&arr_bytes, 4));

    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![("t", &[4, 4], 0, 0)];
    let buf = build_gguf_buffer(&kv, 5, &tensors, 0);
    let f = GgufFile::from_buffer(&buf).unwrap();

    // 对每个 KV 序列化，核对类型标签与值
    for (key, val) in &f.kv {
        let j = value_to_json(val, Some(1000));
        match key.as_str() {
            "k.u32" => assert_eq!(j, json!(7)),
            "k.str" => assert_eq!(j, json!("hello")),
            "k.bool" => assert_eq!(j, json!(true)),
            "k.f32" => assert_eq!(j.as_f64(), Some(3.5)),
            "k.f32arr" => {
                assert_eq!(j["element_type"], "float32");
                assert_eq!(j["count"], 4);
                let shown = j["value"].as_array().unwrap();
                assert_eq!(shown.len(), 4);
                assert_eq!(shown[2], json!(1.0));
            }
            _ => panic!("unexpected key {key}"),
        }
    }
}

fn string_value_bytes(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut v = (b.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(b);
    v
}

fn chunks(bytes: &[u8], chunk: usize) -> Vec<Vec<u8>> {
    bytes.chunks(chunk).map(|c| c.to_vec()).collect()
}
