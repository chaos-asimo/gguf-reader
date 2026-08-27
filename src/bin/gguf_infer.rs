//! GGUF LLM 推理 CLI。
//!
//! 加载 GGUF 模型文件，执行文本生成（支持流式输出）。
//! Windows 下通过宽字符控制台 API 读写，避免管道场景 GBK/UTF-8 乱码。

#[path = "../console_wide.rs"]
mod console_wide;

use clap::{ArgAction, Parser};
use gguf::infer::sampler::SamplerConfig;
use gguf::infer::Engine;
use gguf::GgufError;
use std::io::Write;
use std::process::ExitCode;

/// GGUF LLM 推理引擎 CLI — 加载模型并执行文本生成。
#[derive(Parser, Debug)]
#[command(
    name = "gguf-infer",
    version,
    about = "GGUF 大模型推理引擎（llama / qwen2 / mistral）"
)]
struct Args {
    /// GGUF 模型文件路径
    path: String,

    /// 输入 prompt（缺省从 stdin 读取）
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,

    /// 最大生成 token 数
    #[arg(short = 'n', long, default_value_t = 512)]
    max_tokens: usize,

    /// 温度（0 = 贪心）
    #[arg(short = 't', long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-K（0 = 禁用）
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-P 阈值（1.0 = 禁用）
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Min-P 相对概率阈值（0.0 = 禁用）
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,

    /// 重复惩罚（1.0 = 禁用）
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// 随机种子（0 = 系统随机）
    #[arg(short = 's', long, default_value_t = 0)]
    seed: u64,

    /// 强制贪心解码（忽略 temperature/top-k/top-p）
    #[arg(long)]
    greedy: bool,

    /// 关闭流式输出，一次性返回完整文本
    ///
    /// 默认流式（逐 token 打印）；传 `--no-stream` 关闭流式。
    #[arg(long, action = ArgAction::SetTrue)]
    no_stream: bool,

    /// 打印统计信息（token 数、耗时、速度）
    #[arg(short, long)]
    verbose: bool,

    /// 交互问答模式：加载模型后循环读取输入，上下文持续累积
    ///
    /// 输入 `:reset` 清空上下文重新对话；`:quit` 退出。
    #[arg(long)]
    chat: bool,
}

fn main() -> ExitCode {
    set_utf8_console();
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(exit_code_for(&e))
        }
    }
}

/// Windows 下把控制台输出代码页切到 UTF-8（65001），
/// 避免真实 PowerShell/cmd 窗口里 GBK(936) 解码 UTF-8 中文变乱码。
///
/// - 仅在 stdout/stderr 为真实终端时生效（`is_terminal()`），
///   管道重定向场景下调用会改变控制台代码页，导致 PowerShell
///   管道接收端按错误代码页解码 UTF-8 字节 → 更严重的乱码。
/// - 管道场景的中文显示由调用方负责：
///   `[Console]::OutputEncoding = [Text.Encoding]::UTF8`
/// - 非 Windows 平台 no-op（POSIX 默认 UTF-8）。
#[cfg(windows)]
fn set_utf8_console() {
    use std::io::IsTerminal;
    use windows_sys::Win32::System::Console::SetConsoleOutputCP;
    if std::io::stdout().is_terminal() || std::io::stderr().is_terminal() {
        unsafe {
            let _ = SetConsoleOutputCP(65001); // CP_UTF8
        }
    }
}

#[cfg(not(windows))]
fn set_utf8_console() {}

fn exit_code_for(e: &GgufError) -> u8 {
    match e {
        GgufError::Io(_) => 1,
        GgufError::InvalidMagic(_) => 2,
        GgufError::UnsupportedVersion(_) => 3,
        GgufError::UnsupportedArchitecture(_) => 4,
        GgufError::TokenizerError(_) => 5,
        GgufError::MissingTensor { .. } => 6,
        _ => 7,
    }
}

fn run(args: &Args) -> Result<(), GgufError> {
    // 交互模式下 prompt 在 chat 循环里逐行读取，这里不需要预读
    let prompt = if args.chat {
        String::new()
    } else {
        // 读取 prompt
        let p = match &args.prompt {
            Some(p) => p.trim().to_string(),
            None => {
                eprintln!("(从 stdin 读取 prompt)");
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin().lock(), &mut buf)?;
                buf.trim().to_string()
            }
        };
        if p.is_empty() {
            return Err(GgufError::InferenceError("prompt 为空".into()));
        }
        p
    };

    // 加载模型
    eprintln!("加载模型: {}", args.path);
    let t0 = std::time::Instant::now();
    let file = gguf::GgufFile::from_path(&args.path)?;

    // 构建采样配置
    let cfg = if args.greedy {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: args.seed,
        }
    } else {
        SamplerConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            repeat_penalty: args.repeat_penalty,
            seed: args.seed,
        }
    };

    // 构建引擎
    let mut engine = Engine::new(&file, cfg)?;
    let load_time = t0.elapsed();
    eprintln!(
        "  架构={}  层数={}  隐藏维度={}  词表={}  大小={}  耗时={}",
        file.architecture().unwrap_or("?"),
        engine.hparams().n_layers,
        engine.hparams().embed_dim,
        engine.hparams().vocab_size,
        human_size(file.file_size),
        format_duration(load_time)
    );
    eprintln!(
        "引擎就绪 (embed={}, heads={}, kv_heads={}, ffn={}, ctx={})",
        engine.hparams().embed_dim,
        engine.hparams().n_heads,
        engine.hparams().n_kv_heads,
        engine.hparams().ffn_dim,
        engine.hparams().context_length,
    );

    // 交互问答模式
    if args.chat {
        return run_chat(&mut engine, args);
    }

    // 生成
    let t1 = std::time::Instant::now();
    if !args.no_stream {
        let mut out = std::io::stdout().lock();
        let result = engine.generate(&prompt, args.max_tokens, move |_id, text| {
            let _ = std::io::Write::write_all(&mut out, text.as_bytes());
            let _ = std::io::Write::flush(&mut out);
        });
        if args.verbose {
            let elapsed = t1.elapsed();
            eprintln!("\n生成完成, 耗时 {}", format_duration(elapsed));
        }
        result.map(|_| ())
    } else {
        let s = engine.complete(&prompt, args.max_tokens)?;
        print!("{s}");
        let _ = std::io::stdout().flush();
        if args.verbose {
            let elapsed = t1.elapsed();
            eprintln!("\n生成完成, 耗时 {}", format_duration(elapsed));
        }
        Ok(())
    }
}

/// 人类可读大小。
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = bytes as f64;
    let mut idx = 0;
    while size >= 1024.0 && idx < UNITS.len() - 1 {
        size /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{} B", bytes)
    } else {
        format!("{size:.1} {}", UNITS[idx])
    }
}

/// 交互问答循环：每轮读取用户输入，流式输出模型回复，上下文持续累积。
///
/// Windows 控制台场景用宽字符 API（ReadConsoleW/WriteConsoleW）绕开代码页；
/// 管道/重定向场景回退到字节级 I/O（UTF-8），依赖调用方设置正确编码。
fn run_chat(engine: &mut Engine, args: &Args) -> Result<(), GgufError> {
    use std::io::{BufRead, Write};

    // 检测 I/O 模式
    let wide_input = console_wide::stdin_is_console();
    let wide_output = console_wide::stdout_is_console();

    // 启动提示
    let banner = "\n===== 交互问答模式 =====\n  输入问题后回车发送，模型回复将流式输出\n  特殊命令: :reset 清空上下文  :quit 退出\n";
    if wide_output {
        let _ = console_wide::write_err_wide(banner);
    } else {
        eprint!("{banner}");
    }

    loop {
        // 提示符
        if wide_output {
            let _ = console_wide::write_wide("> ");
        } else {
            print!("> ");
            let _ = std::io::stdout().flush();
        }

        // 读取一行输入
        let input = if wide_input {
            match console_wide::read_line_wide() {
                Ok(Some(s)) => s,
                Ok(None) => break, // EOF
                Err(_) => {
                    // 宽字符读取失败，回退字节级
                    let mut buf = String::new();
                    match std::io::stdin().lock().read_line(&mut buf) {
                        Ok(0) => break,
                        Ok(_) => buf,
                        Err(e) => {
                            eprintln!("\n读取输入错误: {e}");
                            break;
                        }
                    }
                }
            }
        } else {
            let mut buf = String::new();
            match std::io::stdin().lock().read_line(&mut buf) {
                Ok(0) => break, // EOF
                Ok(_) => buf,
                Err(e) => {
                    eprintln!("\n读取输入错误: {e}");
                    break;
                }
            }
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == ":quit" || input == ":q" {
            if wide_output {
                let _ = console_wide::write_err_wide("退出。\n");
            } else {
                eprintln!("退出。");
            }
            break;
        }
        if input == ":reset" {
            engine.reset();
            if wide_output {
                let _ = console_wide::write_err_wide("[上下文已清空]\n");
            } else {
                eprintln!("[上下文已清空]");
            }
            continue;
        }

        let t0 = std::time::Instant::now();
        let result = if wide_output {
            // 宽字符流式输出
            engine.chat(input, args.max_tokens, |_id, text| {
                let _ = console_wide::write_wide(text);
            })
        } else {
            // 字节级流式输出（管道场景，UTF-8）
            let mut out = std::io::stdout().lock();
            let r = engine.chat(input, args.max_tokens, |_id, text| {
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            });
            let _ = out.write_all(b"\n");
            let _ = out.flush();
            r
        };
        if wide_output {
            let _ = console_wide::write_wide("\n");
        } else {
            println!();
        }

        // 状态行（stderr）
        let status_line = match &result {
            Ok(_) => {
                let ctx_used = engine.model_mut().cache_len();
                let ctx_limit = engine.hparams().context_length as usize;
                if args.verbose {
                    format!("  [耗时 {}, ctx {}/{}]\n", format_duration(t0.elapsed()), ctx_used, ctx_limit)
                } else {
                    format!("  [耗时 {}]\n", format_duration(t0.elapsed()))
                }
            }
            Err(e) => {
                let mut s = format!("  错误: {e}\n");
                if e.to_string().contains("上下文超出") {
                    s.push_str("  提示: 输入 :reset 清空上下文后重新提问\n");
                }
                s
            }
        };
        if wide_output {
            let _ = console_wide::write_err_wide(&status_line);
        } else {
            eprint!("{status_line}");
        }

        if result.is_err() && !wide_output {
            // 字节模式下 Err 时不 break，继续循环（用户可 :reset）
        }
    }
    Ok(())
}

/// 格式化时间。
fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.0}ms", d.as_secs_f64() * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.2}s")
    } else {
        format!("{:.1}min", secs / 60.0)
    }
}
