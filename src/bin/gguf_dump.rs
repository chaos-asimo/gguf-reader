use clap::Parser;
use gguf::GgufError;
use serde_json::{json, Value as J};
use std::process::ExitCode;

/// 查看 GGUF 大模型文件的元数据（文件头、KV 键值、张量描述符）。
#[derive(Parser, Debug)]
#[command(name = "gguf-dump", version, about)]
struct Args {
    /// GGUF 文件路径
    path: String,

    /// 以 JSON 格式输出
    #[arg(short = 'j', long)]
    json: bool,

    /// JSON 美化输出（仅对 --json 有意义）
    #[arg(long)]
    pretty: bool,

    /// 文本模式下显示全部张量（默认截断前 50）
    #[arg(long)]
    tensors_all: bool,

    /// 文本模式下 KV 显示上限（默认 200）
    #[arg(short = 'm', long, default_value_t = 200)]
    max_kv: usize,

    /// 仅显示指定键的 KV 值（可多次指定）
    #[arg(short = 'k', long = "key")]
    keys: Vec<String>,

    /// 仅显示文件摘要
    #[arg(long)]
    summary_only: bool,

    /// 静默模式：不显示张量与 KV，仅摘要
    #[arg(short = 'q', long)]
    quiet: bool,
}

/// JSON 输出中数组截断阈值。
const JSON_ARRAY_LIMIT: usize = 1000;
/// 文本输出中张量默认截断数。
const TENSOR_TEXT_LIMIT: usize = 50;

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(exit_code_for(&e))
        }
    }
}

/// 将 GgufError 映射到 CLI 退出码。
fn exit_code_for(e: &GgufError) -> u8 {
    match e {
        GgufError::Io(_) => 1,
        GgufError::InvalidMagic(_) => 2,
        GgufError::UnsupportedVersion(_) => 3,
        _ => 4,
    }
}

fn run(args: &Args) -> Result<(), GgufError> {
    let file = gguf::GgufFile::from_path(&args.path)?;

    if args.json {
        print_json(&file, args);
    } else {
        print_text(&file, args);
    }
    Ok(())
}

// ---------------- 文本输出 ----------------

fn print_text(file: &gguf::GgufFile, args: &Args) {
    let arch = file.architecture().unwrap_or("(unknown)");
    let name = file.model_name().unwrap_or("(unnamed)");

    println!("GGUF File: {}", args.path);
    println!("=====================");
    println!("Size:            {}", human_size(file.file_size));
    println!("Version:         {}", file.header.version);
    println!("Tensors:         {}", with_commas(file.header.n_tensors));
    println!("KV pairs:        {}", with_commas(file.header.n_kv));
    println!("Alignment:       {}", file.alignment);
    println!(
        "Data offset:     {} bytes ({})",
        with_commas(file.data_offset),
        human_size(file.data_offset)
    );
    println!("Architecture:    {arch}");
    println!("Model name:      {name}");

    if args.quiet || args.summary_only {
        return;
    }

    // KV 元数据
    println!(
        "\nKey-Value Metadata (showing {} of {}):",
        file.kv.len(),
        file.header.n_kv
    );
    let limit = if args.keys.is_empty() {
        args.max_kv
    } else {
        usize::MAX
    };
    let mut shown = 0usize;
    for (k, v) in &file.kv {
        if !args.keys.is_empty() && !args.keys.iter().any(|key| key == k) {
            continue;
        }
        if shown >= limit {
            println!("  ... ({} more keys)", file.kv.len() - shown);
            break;
        }
        println!("  {:<38} {:<14} {}", k, v.value_type(), v.display());
        shown += 1;
    }
    if shown == 0 {
        println!("  (no matching keys)");
    }

    // 张量列表
    let limit = if args.tensors_all {
        file.tensors.len()
    } else {
        TENSOR_TEXT_LIMIT.min(file.tensors.len())
    };
    println!(
        "\nTensors (showing first {limit} of {}):",
        file.header.n_tensors
    );
    if limit > 0 {
        print_tensor_table(&file.tensors[..limit]);
    }
}

fn print_tensor_table(tensors: &[gguf::TensorInfo]) {
    let header = format!(
        "{:<40} {:<24} {:<14} {:>16} {:>18}",
        "NAME", "SHAPE", "TYPE", "OFFSET", "SIZE"
    );
    println!("  {header}");
    for t in tensors {
        let shape = format_shape(&t.shape);
        let size = t
            .est_data_size()
            .map(with_commas)
            .unwrap_or_else(|| "—".to_string());
        println!(
            "  {:<40} {:<24} {:<14} {:>16} {:>18}",
            truncate(&t.name, 40),
            shape,
            t.dtype,
            with_commas(t.offset),
            size
        );
    }
}

fn format_shape(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "()".to_string();
    }
    let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", inner.join(", "))
}

// ---------------- JSON 输出 ----------------

fn print_json(file: &gguf::GgufFile, args: &Args) {
    let kv_obj = build_kv_json(file, args);
    let tensors_arr: Vec<J> = file
        .tensors
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "shape": t.shape,
                "dtype": t.dtype.to_string(),
                "offset": t.offset,
                "num_elements": t.num_elements(),
            })
        })
        .collect();

    let obj = json!({
        "file": args.path,
        "size": file.file_size,
        "header": {
            "magic": format!("{:08x}", file.header.magic),
            "version": file.header.version,
            "tensors": file.header.n_tensors,
            "kv_pairs": file.header.n_kv,
        },
        "alignment": file.alignment,
        "data_offset": file.data_offset,
        "architecture": file.architecture(),
        "model_name": file.model_name(),
        "kv": kv_obj,
        "tensors": tensors_arr,
    });

    if args.pretty {
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
    } else {
        println!("{}", serde_json::to_string(&obj).unwrap_or_default());
    }
}

fn build_kv_json(file: &gguf::GgufFile, args: &Args) -> J {
    let mut map = serde_json::Map::new();
    for (k, v) in &file.kv {
        if !args.keys.is_empty() && !args.keys.iter().any(|key| key == k) {
            continue;
        }
        let val = gguf::value_to_json(v, Some(JSON_ARRAY_LIMIT));
        let entry = json!({
            "type": v.value_type().to_string(),
            "value": val,
        });
        map.insert(k.clone(), entry);
    }
    J::Object(map)
}

// ---------------- 格式化工具 ----------------

/// 人类可读大小（B/KB/MB/GB/TB）。
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = bytes as f64;
    let mut idx = 0;
    while size >= 1024.0 && idx < UNITS.len() - 1 {
        size /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{size:.2} {}", UNITS[idx])
    }
}

/// 千分位逗号分隔。
fn with_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// 截断字符串到指定显示宽度（按字符数近似）。
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{t}…")
    }
}
