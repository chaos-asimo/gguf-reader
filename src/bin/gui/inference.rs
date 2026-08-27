//! 后台推理线程与消息通道。
//!
//! UI 线程通过 `UiCommand` 发送命令，推理线程通过 `InferMsg` 返回结果。
//! `Engine` 仅在推理线程中持有和使用（`&mut self` 方法）。
//! 停止机制通过 `Arc<AtomicBool>` 实现。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use gguf::infer::sampler::SamplerConfig;
use gguf::infer::Engine;
use gguf::{GgufFile, GgufResult};

/// UI → 推理线程命令
#[derive(Debug)]
pub enum UiCommand {
    /// 加载模型
    LoadModel {
        path: String,
        sampler: SamplerConfig,
    },
    /// 单轮 prompt 补全
    Prompt {
        text: String,
        max_tokens: usize,
        sampler: SamplerConfig,
        stream: bool,
    },
    /// 多轮对话
    Chat {
        text: String,
        max_tokens: usize,
        sampler: SamplerConfig,
    },
    /// 重置对话上下文
    Reset,
    /// 退出
    Quit,
}

/// 推理线程 → UI 消息
#[derive(Debug, Clone)]
pub enum InferMsg {
    /// 模型加载完成
    ModelLoaded {
        name: String,
        arch: String,
        model_name: String,
        layers: u32,
        embed_dim: u32,
        vocab_size: u32,
        size_mb: f64,
        load_ms: u128,
        tensor_count: usize,
        kv_count: usize,
        gguf_version: u32,
        alignment: u32,
        data_offset: u64,
        file_size: u64,
        ctx_limit: usize,
        kv_data: Vec<(String, String, String)>,
        tensor_data: Vec<(String, String, String)>,
    },
    /// 模型加载失败
    LoadError { message: String },
    /// 流式 token
    Token { id: u32, text: String },
    /// 一轮生成完成
    Done {
        full_text: String,
        elapsed_ms: u128,
        ctx_len: usize,
        ctx_limit: usize,
        token_count: usize,
    },
    /// 生成出错
    Error { message: String },
    /// 已停止
    Stopped,
    /// 已重置
    ResetDone,
}

/// 推理线程句柄，持有命令发送端和停止标志。
pub struct InferenceHandle {
    pub cmd_tx: mpsc::Sender<UiCommand>,
    pub msg_rx: mpsc::Receiver<InferMsg>,
    pub stop_flag: Arc<AtomicBool>,
}

/// 创建推理线程，返回句柄。
pub fn spawn_inference() -> InferenceHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<UiCommand>();
    let (msg_tx, msg_rx) = mpsc::channel::<InferMsg>();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = Arc::clone(&stop_flag);

    std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            inference_loop(cmd_rx, msg_tx, stop_flag_clone);
        })
        .expect("启动推理线程失败");

    InferenceHandle {
        cmd_tx,
        msg_rx,
        stop_flag,
    }
}

/// 封装已加载的模型：GgufFile + Engine。
///
/// `Engine<'_>` 借用 `GgufFile`，两者必须在同一 struct 中共同存活。
/// 由于 Rust 不允许 struct 字段借用同一 struct 的另一个字段，
/// 这里通过 `#[allow(invalid_reference_casting)]` 将引用提升为 `'static`。
/// 安全性保证：`LoadedModel` 仅在推理线程中创建和使用，
/// 替换（drop 旧值）时 `Engine` 先被 drop（结构体字段 drop 顺序按声明顺序，
/// `engine` 声明在前先 drop），然后 `file` 才被 drop。
#[allow(dead_code)]
struct LoadedModel {
    /// Engine 声明在前，确保 drop 顺序：engine → file
    engine: Engine<'static>,
    file: GgufFile,
}

impl LoadedModel {
    fn new(file: GgufFile, sampler: SamplerConfig) -> GgufResult<Self> {
        let file_ptr = &file as *const GgufFile;
        // SAFETY: file 被 move 到 Self 中，其内存地址在整个 LoadedModel 生命周期内有效。
        // Engine 仅在推理线程中使用，LoadedModel 替换时 engine 先 drop（字段声明顺序）。
        let static_file = unsafe { &*file_ptr };
        let engine = Engine::new(static_file, sampler)?;
        Ok(Self { engine, file })
    }
}

fn inference_loop(
    cmd_rx: mpsc::Receiver<UiCommand>,
    msg_tx: mpsc::Sender<InferMsg>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut loaded: Option<LoadedModel> = None;

    for cmd in cmd_rx {
        let should_quit = matches!(cmd, UiCommand::Quit);
        stop_flag.store(false, Ordering::Relaxed);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_command(&cmd, &mut loaded, &msg_tx, &stop_flag);
        }));
        if result.is_err() {
            let _ = msg_tx.send(InferMsg::Error {
                message: "推理线程 panic".into(),
            });
        }
        if should_quit {
            break;
        }
    }
}

fn handle_command(
    cmd: &UiCommand,
    loaded: &mut Option<LoadedModel>,
    msg_tx: &mpsc::Sender<InferMsg>,
    stop_flag: &Arc<AtomicBool>,
) {
    match cmd {
        UiCommand::LoadModel { path, sampler } => {
            let start = std::time::Instant::now();
            match LoadedModel::new(GgufFile::from_path(path).unwrap_or_else(|e| {
                let _ = msg_tx.send(InferMsg::LoadError {
                    message: format!("{e}"),
                });
                panic!("GgufFile load failed")
            }), sampler.clone()) {
                Ok(model) => {
                    let eng = &model.engine;
                    let file = &model.file;
                    let hp = eng.hparams();
                    let name = file
                        .model_name()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            std::path::Path::new(path)
                                .file_name()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_default()
                        });
                    let arch = hp.arch.clone();
                    let model_name = file
                        .model_name()
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let ctx_limit = hp.context_length as usize;

                    let kv_data: Vec<(String, String, String)> = file
                        .kv
                        .iter()
                        .map(|(k, v)| {
                            (
                                k.clone(),
                                format!("{}", v.value_type()),
                                value_to_string(v),
                            )
                        })
                        .collect();
                    let tensor_data: Vec<(String, String, String)> = file
                        .tensors
                        .iter()
                        .map(|t| {
                            (
                                t.name.clone(),
                                format_shape(&t.shape),
                                format!("{}", t.dtype),
                            )
                        })
                        .collect();

                    let _ = msg_tx.send(InferMsg::ModelLoaded {
                        name,
                        arch,
                        model_name,
                        layers: hp.n_layers,
                        embed_dim: hp.embed_dim,
                        vocab_size: hp.vocab_size,
                        size_mb: file.file_size as f64 / (1024.0 * 1024.0),
                        load_ms: start.elapsed().as_millis(),
                        tensor_count: file.tensors.len(),
                        kv_count: file.kv.len(),
                        gguf_version: file.header.version,
                        alignment: file.alignment,
                        data_offset: file.data_offset,
                        file_size: file.file_size,
                        ctx_limit,
                        kv_data,
                        tensor_data,
                    });

                    // 替换旧模型（旧 LoadedModel 在此 drop）
                    *loaded = Some(model);
                }
                Err(e) => {
                    let _ = msg_tx.send(InferMsg::LoadError {
                        message: format!("{e}"),
                    });
                }
            }
        }

        UiCommand::Prompt {
            text,
            max_tokens,
            sampler,
            stream,
        } => {
            let Some(model) = loaded.as_mut() else {
                let _ = msg_tx.send(InferMsg::Error {
                    message: "模型未加载".into(),
                });
                return;
            };
            let eng = &mut model.engine;
            eng.set_sampler_config(sampler.clone());

            let start = std::time::Instant::now();
            let ctx_limit = eng.hparams().context_length as usize;

            if *stream {
                let stop = stop_flag.clone();
                let result = eng.generate_cancellable(text, *max_tokens, |id, txt| {
                    let _ = msg_tx.send(InferMsg::Token {
                        id,
                        text: txt.to_string(),
                    });
                    !stop.load(Ordering::Relaxed)
                });
                send_result(result, msg_tx, start, ctx_limit, *max_tokens);
            } else {
                let result = eng.complete(text, *max_tokens);
                send_result(result, msg_tx, start, ctx_limit, *max_tokens);
            }
        }

        UiCommand::Chat {
            text,
            max_tokens,
            sampler,
        } => {
            let Some(model) = loaded.as_mut() else {
                let _ = msg_tx.send(InferMsg::Error {
                    message: "模型未加载".into(),
                });
                return;
            };
            let eng = &mut model.engine;
            eng.set_sampler_config(sampler.clone());

            let start = std::time::Instant::now();
            let ctx_limit = eng.hparams().context_length as usize;

            let stop = stop_flag.clone();
            let result = eng.chat_cancellable(text, *max_tokens, |id, txt| {
                let _ = msg_tx.send(InferMsg::Token {
                    id,
                    text: txt.to_string(),
                });
                !stop.load(Ordering::Relaxed)
            });
            match &result {
                Ok(full_text) => {
                    let ctx_len = eng.model_mut().cache_len();
                    let _ = msg_tx.send(InferMsg::Done {
                        full_text: full_text.clone(),
                        elapsed_ms: start.elapsed().as_millis(),
                        ctx_len,
                        ctx_limit,
                        token_count: 0,
                    });
                }
                Err(e) => {
                    let _ = msg_tx.send(InferMsg::Error {
                        message: format!("{e}"),
                    });
                }
            }
        }

        UiCommand::Reset => {
            if let Some(model) = loaded.as_mut() {
                model.engine.reset();
                let _ = msg_tx.send(InferMsg::ResetDone);
            }
        }

        UiCommand::Quit => {}
    }
}

fn send_result(
    result: GgufResult<String>,
    msg_tx: &mpsc::Sender<InferMsg>,
    start: std::time::Instant,
    ctx_limit: usize,
    max_tokens: usize,
) {
    match result {
        Ok(full_text) => {
            let _ = msg_tx.send(InferMsg::Done {
                full_text,
                elapsed_ms: start.elapsed().as_millis(),
                ctx_len: 0,
                ctx_limit,
                token_count: max_tokens,
            });
        }
        Err(e) => {
            let _ = msg_tx.send(InferMsg::Error {
                message: format!("{e}"),
            });
        }
    }
}

fn format_shape(shape: &[u64]) -> String {
    let parts: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn value_to_string(v: &gguf::GgufValue) -> String {
    use gguf::GgufValue;
    match v {
        GgufValue::U8(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::F32(x) => format!("{x:.4}"),
        GgufValue::Bool(x) => x.to_string(),
        GgufValue::String(s) => s.clone(),
        GgufValue::U64(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::F64(x) => format!("{x:.4}"),
        GgufValue::Array(arr) => {
            let items: Vec<String> = arr
                .data
                .iter()
                .take(8)
                .map(value_to_string)
                .collect();
            let suffix = if arr.data.len() > 8 { ", ..." } else { "" };
            format!("[{}{}]", items.join(", "), suffix)
        }
    }
}
