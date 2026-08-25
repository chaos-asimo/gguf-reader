//! CLI 集成测试：通过运行 gguf-dump 二进制验证文本/JSON 输出与退出码。

mod common;

use common::*;
use std::process::Command;
use std::string::ToString;

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

/// --quiet 仅输出摘要，不含 KV 列表与张量列表。
#[test]
fn test_cli_quiet() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--quiet")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Architecture"), "quiet 应含摘要");
    assert!(
        !stdout.contains("Key-Value Metadata"),
        "quiet 不应含 KV 列表"
    );
    assert!(!stdout.contains("Tensors (showing"), "quiet 不应含张量列表");
    assert!(!stdout.contains("token_embd.weight"), "quiet 不应含张量名");
}

/// 短选项 -q 等价于 --quiet。
#[test]
fn test_cli_quiet_short_flag() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("-q")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("Key-Value Metadata"));
}

/// --tensors-all 显示全部张量（默认截断前 50）。
#[test]
fn test_cli_tensors_all() {
    // 构造 60 个张量
    let kv = Vec::new();
    let dims: [i64; 2] = [1, 1];
    let tensors: Vec<(&str, &[i64], i32, u64)> = (0..60)
        .map(|i| {
            (
                format!("t{i:02}").leak() as &str,
                &dims as &[i64],
                0i32,
                i as u64 * 4,
            )
        })
        .collect();
    let buf = build_gguf_buffer(&kv, 0, &tensors, 0);
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();

    // 默认：仅前 50
    let out = Command::new(&bin).arg(&path).output().expect("run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("showing first 50 of 60"), "默认应截断 50");
    assert!(!stdout.contains("t59"), "默认不应含第 60 个张量");

    // --tensors-all：全部 60
    let out = Command::new(&bin)
        .arg("--tensors-all")
        .arg(&path)
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("showing first 60 of 60"),
        "tensors-all 应显示全部"
    );
    assert!(stdout.contains("t59"), "tensors-all 应含第 60 个张量");
}

/// --max-kv 限制 KV 显示数量（默认 200）。
#[test]
fn test_cli_max_kv() {
    // 构造 5 个 KV，限制显示 2 个
    let mut kv = Vec::new();
    for i in 0..5 {
        write_scalar_kv(&mut kv, &format!("k{i}"), 4, &(i as u32).to_le_bytes());
    }
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 5, &tensors, 0);
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();

    let out = Command::new(&bin)
        .arg("--max-kv")
        .arg("2")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // 前 2 个键应显示
    assert!(stdout.contains("k0"));
    assert!(stdout.contains("k1"));
    // 第 3 个及以后不应显示，但应有 "more keys" 提示
    assert!(!stdout.contains("k2"));
    assert!(stdout.contains("more keys"));
}

/// 短选项 -m 等价于 --max-kv。
#[test]
fn test_cli_max_kv_short_flag() {
    let mut kv = Vec::new();
    for i in 0..5 {
        write_scalar_kv(&mut kv, &format!("k{i}"), 4, &(i as u32).to_le_bytes());
    }
    let tensors: Vec<(&str, &[i64], i32, u64)> = vec![];
    let buf = build_gguf_buffer(&kv, 5, &tensors, 0);
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();

    let out = Command::new(&bin)
        .arg("-m")
        .arg("1")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("k0"));
    assert!(!stdout.contains("k1"));
}

/// 多次 --key 过滤：仅显示匹配的多个键。
#[test]
fn test_cli_multiple_key_filter() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--key")
        .arg("general.architecture")
        .arg("--key")
        .arg("llama.block_count")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("general.architecture"));
    assert!(stdout.contains("llama.block_count"));
    // 未匹配的键不应出现
    assert!(!stdout.contains("general.name"));
}

/// --help 输出包含主要选项说明，退出码 0。
#[test]
fn test_cli_help() {
    let bin = bin_path();
    let out = Command::new(&bin).arg("--help").output().expect("run");
    assert!(out.status.success(), "--help 应退出 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--json"));
    assert!(stdout.contains("--key"));
    assert!(stdout.contains("--max-kv"));
    assert!(stdout.contains("--summary-only"));
    assert!(stdout.contains("--quiet"));
    assert!(stdout.contains("--tensors-all"));
    assert!(stdout.contains("--pretty"));
}

/// --version 输出版本号，退出码 0。
#[test]
fn test_cli_version() {
    let bin = bin_path();
    let out = Command::new(&bin).arg("--version").output().expect("run");
    assert!(out.status.success(), "--version 应退出 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Cargo.toml version = "0.1.0"
    assert!(stdout.contains("0.1.0"), "应含版本号，实际: {stdout}");
}

/// JSON 模式下 --key 过滤同样生效。
#[test]
fn test_cli_json_key_filter() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("--json")
        .arg("--key")
        .arg("llama.block_count")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    // 仅含被过滤的键
    assert!(v["kv"].get("llama.block_count").is_some());
    assert!(v["kv"].get("general.architecture").is_none());
    assert!(v["kv"].get("general.name").is_none());
    // header / tensors 仍完整
    assert_eq!(v["header"]["version"], 3);
    assert_eq!(v["tensors"].as_array().unwrap().len(), 2);
}

/// 短选项 -j 等价于 --json。
#[test]
fn test_cli_json_short_flag() {
    let buf = sample_gguf();
    let (_dir, path) = write_temp_gguf(&buf);
    let bin = bin_path();
    let out = Command::new(&bin)
        .arg("-j")
        .arg(&path)
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(v["architecture"], "llama");
}

/// 无参数运行：clap 报错，退出码非 0。
#[test]
fn test_cli_no_args() {
    let bin = bin_path();
    let out = Command::new(&bin).output().expect("run");
    assert!(!out.status.success(), "无参数应失败");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("path") || stderr.contains("required"),
        "应提示缺少 path"
    );
}
