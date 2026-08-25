//! CLI 集成测试：通过运行 gguf-dump 二进制验证文本/JSON 输出与退出码。

mod common;

use common::*;
use std::process::Command;

/// 获取当前目录下的 target 中 gguf-dump 可执行文件路径。
fn bin_path() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap();
    // .../target/debug/deps/cli-<hash> → .../target/debug/
    p.pop();
    p.pop();
    p.push("gguf-dump.exe");
    p
}

/// 写入一个临时 GGUF 文件，返回路径字符串。
fn write_temp_gguf(content: &[u8]) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    std::fs::write(&path, content).unwrap();
    (dir, path.to_string_lossy().to_string())
}

/// 构造一个含少量 KV 与张量的合法 GGUF 缓冲。
fn sample_gguf() -> Vec<u8> {
    let mut kv = Vec::new();
    // general.architecture = "llama" (string)
    write_scalar_kv(&mut kv, "general.architecture", 8, &str_bytes("llama"));
    // general.name = "TestModel" (string)
    write_scalar_kv(&mut kv, "general.name", 8, &str_bytes("TestModel"));
    // llama.block_count = 32 (u32)
    write_scalar_kv(&mut kv, "llama.block_count", 4, &32u32.to_le_bytes());
    // general.alignment = 32 (u32)
    write_scalar_kv(&mut kv, "general.alignment", 4, &32u32.to_le_bytes());

    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![
        ("token_embd.weight", &[128, 4096], 30, 0),
        ("output.weight", &[4096], 0, 1048576),
    ];
    build_gguf_buffer(&kv, 4, &tensors, 2048)
}

fn str_bytes(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut v = (b.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(b);
    v
}

#[test]
fn test_cli_text_output() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg(&path)
        .output()
        .expect("run gguf-dump");
    assert!(out.status.success(), "exit not success: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("llama"), "should contain architecture");
    assert!(stdout.contains("TestModel"), "should contain model name");
    assert!(
        stdout.contains("token_embd.weight"),
        "should contain tensor name"
    );
    assert!(stdout.contains("KV pairs"), "should contain summary");
    assert!(stdout.contains("block_count"), "should contain kv key");
}

#[test]
fn test_cli_json_output() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--json")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // 验证是合法 JSON
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(v["header"]["version"], 3);
    assert_eq!(v["architecture"], "llama");
    assert_eq!(v["model_name"], "TestModel");
    assert_eq!(v["kv"]["general.architecture"]["value"], "llama");
    assert_eq!(v["kv"]["llama.block_count"]["value"], 32);
    assert_eq!(v["tensors"][0]["name"], "token_embd.weight");
    assert_eq!(v["tensors"][0]["shape"][0], 128);
    assert_eq!(v["tensors"][0]["shape"][1], 4096);
    assert_eq!(v["tensors"][0]["dtype"], "BF16");
}

#[test]
fn test_cli_json_pretty() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--json")
        .arg("--pretty")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // pretty 输出应包含换行与缩进
    assert!(stdout.contains('\n'));
    assert!(stdout.contains("  "));
}

#[test]
fn test_cli_key_filter() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--key")
        .arg("llama.block_count")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("llama.block_count"));
    // 被过滤的键不应出现
    assert!(!stdout.contains("general.name"));
}

#[test]
fn test_cli_summary_only() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--summary-only")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Architecture"));
    // 不应出现 KV 列表与张量列表
    assert!(!stdout.contains("Key-Value Metadata"));
    assert!(!stdout.contains("token_embd.weight"));
}

#[test]
fn test_cli_exit_code_missing_file() {
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("nonexistent_file_12345.gguf")
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1), "missing file → exit 1");
}

#[test]
fn test_cli_exit_code_bad_magic() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"XXXX");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin).arg(&path).output().expect("run");
    assert_eq!(out.status.code(), Some(2), "bad magic → exit 2");
}

#[test]
fn test_cli_exit_code_bad_version() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&99u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin).arg(&path).output().expect("run");
    assert_eq!(out.status.code(), Some(3), "bad version → exit 3");
}

#[test]
fn test_cli_exit_code_corrupt() {
    // 声称 1 个 KV 但无内容 → 损坏 → exit 4
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&1i64.to_le_bytes());
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin).arg(&path).output().expect("run");
    assert_eq!(out.status.code(), Some(4), "corrupt → exit 4");
}

#[test]
fn test_cli_json_array_truncation() {
    // 构造一个含 1500 个元素的 f32 数组 KV
    let elems: Vec<Vec<u8>> = (0..1500)
        .map(|i| (i as f32).to_le_bytes().to_vec())
        .collect();
    let mut kv = Vec::new();
    write_array_kv(&mut kv, "big.array", 6, &elems);
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 1, &tensors, 0);
    let (_dir, path) = write_temp_gguf(&buf);

    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--json")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    let arr = &v["kv"]["big.array"]["value"];
    assert_eq!(arr["count"], 1500);
    assert_eq!(arr["truncated"], true);
    assert_eq!(arr["total"], 1500);
    // 实际只保留前 1000 项
    assert_eq!(arr["value"].as_array().unwrap().len(), 1000);
}
