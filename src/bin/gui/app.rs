//! 主 App 结构体 + 标签页管理。

use eframe::egui;

use super::inference::{spawn_inference, InferMsg, InferenceHandle, UiCommand};
use super::state::{AppState, ChatMessage, GenStats, ModelSummary, Role};
use super::chat_view::ChatView;
use super::prompt_view::PromptView;
use super::model_view::ModelView;
use super::settings::SettingsPanel;

/// 标签页
#[derive(Clone, Copy, PartialEq)]
pub enum Tab {
    Chat,
    Prompt,
    Model,
}

impl Tab {
    pub fn label(&self) -> &'static str {
        match self {
            Tab::Chat => "对话",
            Tab::Prompt => "Prompt 补全",
            Tab::Model => "模型信息",
        }
    }
}

/// 主应用
pub struct GgufApp {
    pub inference: InferenceHandle,
    pub app_state: AppState,
    pub error_msg: String,
    pub tab: Tab,
    pub settings: SettingsPanel,
    pub chat: ChatView,
    pub prompt: PromptView,
    pub model: ModelView,
    pub model_path: String,
    pub model_name: String,
    pub model_summary: Option<ModelSummary>,
    pub stats: GenStats,
    /// 当前正在流式生成的目标（用于 Token 消息路由）
    pub streaming_target: StreamingTarget,
    /// 流式 token 计数
    pub stream_token_count: usize,
    /// 流式生成开始时间
    pub stream_start: Option<std::time::Instant>,
}

/// 流式 token 路由目标
#[derive(Clone, Copy, PartialEq, Default)]
pub enum StreamingTarget {
    #[default]
    None,
    Chat,
    Prompt,
}

impl GgufApp {
    /// 创建新应用实例。
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let inference = spawn_inference();
        let mut settings = SettingsPanel::default();
        settings.cmd_tx = Some(inference.cmd_tx.clone());
        settings.stop_flag = Some(inference.stop_flag.clone());
        let _ = cc;
        Self {
            inference,
            app_state: AppState::Idle,
            error_msg: String::new(),
            tab: Tab::Chat,
            settings,
            chat: ChatView::default(),
            prompt: PromptView::default(),
            model: ModelView::default(),
            model_path: String::new(),
            model_name: String::new(),
            model_summary: None,
            stats: GenStats::default(),
            streaming_target: StreamingTarget::None,
            stream_token_count: 0,
            stream_start: None,
        }
    }

    /// 构建当前采样配置。
    pub fn current_sampler_config(&self) -> gguf::infer::sampler::SamplerConfig {
        self.settings.sampler_config()
    }

    /// 加载模型。
    pub fn load_model(&mut self, path: String) {
        self.model_path = path.clone();
        self.app_state = AppState::Loading;
        self.error_msg.clear();
        self.streaming_target = StreamingTarget::None;
        let _ = self.inference.cmd_tx.send(UiCommand::LoadModel {
            path,
            sampler: self.current_sampler_config(),
        });
    }

    /// 停止生成。
    pub fn stop_generation(&mut self) {
        self.inference.stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// 发送对话消息。
    pub fn send_chat(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.chat.messages.push(ChatMessage {
            role: Role::User,
            content: text.to_string(),
        });
        self.chat.messages.push(ChatMessage {
            role: Role::Assistant,
            content: String::new(),
        });
        self.streaming_target = StreamingTarget::Chat;
        self.stream_token_count = 0;
        self.stream_start = Some(std::time::Instant::now());
        self.app_state = AppState::Generating;
        let _ = self.inference.cmd_tx.send(UiCommand::Chat {
            text: text.to_string(),
            max_tokens: self.settings.max_tokens,
            sampler: self.current_sampler_config(),
        });
    }

    /// 发送 prompt。
    pub fn send_prompt(&mut self) {
        let text = self.prompt.input_text.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.prompt.output.clear();
        self.streaming_target = if self.prompt.stream {
            StreamingTarget::Prompt
        } else {
            StreamingTarget::None
        };
        self.stream_token_count = 0;
        self.stream_start = Some(std::time::Instant::now());
        self.app_state = AppState::Generating;
        let _ = self.inference.cmd_tx.send(UiCommand::Prompt {
            text,
            max_tokens: self.settings.max_tokens,
            sampler: self.current_sampler_config(),
            stream: self.prompt.stream,
        });
    }

    /// 重置对话。
    pub fn reset_chat(&mut self) {
        self.chat.messages.clear();
        self.chat.ctx_exceeded = false;
        let _ = self.inference.cmd_tx.send(UiCommand::Reset);
    }

    /// 处理推理线程消息。
    pub fn handle_infer_msg(&mut self, msg: InferMsg) {
        match msg {
            InferMsg::ModelLoaded {
                name,
                arch,
                model_name,
                size_mb,
                load_ms,
                tensor_count,
                kv_count,
                gguf_version,
                alignment,
                data_offset,
                file_size,
                ctx_limit,
                kv_data,
                tensor_data,
                embed_dim,
                vocab_size,
                layers: _,
            } => {
                self.app_state = AppState::Ready;
                self.model_name = name.clone();
                self.model_summary = Some(ModelSummary {
                    name: name.clone(),
                    arch,
                    model_name,
                    gguf_version,
                    alignment,
                    data_offset,
                    file_size,
                    tensor_count,
                    kv_count,
                    load_ms,
                });
                self.model.summary = self.model_summary.clone();
                self.model.kv_data = kv_data;
                self.model.tensor_data = tensor_data;
                self.model.ctx_limit = ctx_limit;
                self.model.embed_dim = embed_dim;
                self.model.vocab_size = vocab_size;
                self.stats.ctx_limit = ctx_limit;
                self.settings.stats.ctx_limit = ctx_limit;
                // 重置对话和 prompt
                self.chat = ChatView::default();
                self.prompt = PromptView::default();
                let _ = size_mb;
            }
            InferMsg::LoadError { message } => {
                self.app_state = AppState::Error;
                self.error_msg = message;
            }
            InferMsg::Token { id: _, text } => {
                self.stream_token_count += 1;
                match self.streaming_target {
                    StreamingTarget::Chat => {
                        if let Some(last) = self.chat.messages.last_mut() {
                            if last.role == Role::Assistant {
                                last.content.push_str(&text);
                            }
                        }
                    }
                    StreamingTarget::Prompt => {
                        self.prompt.output.push_str(&text);
                    }
                    StreamingTarget::None => {}
                }
                // 请求重绘
            }
            InferMsg::Done {
                full_text,
                elapsed_ms,
                ctx_len,
                ctx_limit,
                token_count: _,
            } => {
                let elapsed = self
                    .stream_start
                    .map(|s| s.elapsed().as_millis())
                    .unwrap_or(elapsed_ms);
                self.stats = GenStats {
                    elapsed_ms: elapsed,
                    tokens: self.stream_token_count,
                    ctx_len,
                    ctx_limit,
                };
                if self.streaming_target == StreamingTarget::Prompt && !self.prompt.stream {
                    self.prompt.output = full_text;
                }
                self.app_state = AppState::Ready;
                self.streaming_target = StreamingTarget::None;
                self.stream_start = None;
            }
            InferMsg::Error { message } => {
                self.app_state = AppState::Error;
                self.error_msg = message.clone();
                if self.streaming_target == StreamingTarget::Chat {
                    if let Some(last) = self.chat.messages.last_mut() {
                        if last.role == Role::Assistant {
                            last.content.push_str(&format!("\n⚠️ {message}"));
                        }
                    }
                }
                self.streaming_target = StreamingTarget::None;
                self.stream_start = None;
            }
            InferMsg::Stopped => {
                self.app_state = AppState::Ready;
                self.streaming_target = StreamingTarget::None;
                self.stream_start = None;
            }
            InferMsg::ResetDone => {
                self.app_state = AppState::Ready;
                self.stats = GenStats::default();
            }
        }
    }
}

impl eframe::App for GgufApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 批量 drain 消息
        let mut msg_count = 0;
        while let Ok(msg) = self.inference.msg_rx.try_recv() {
            self.handle_infer_msg(msg);
            msg_count += 1;
        }
        if msg_count > 0 {
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.render_toolbar(ui);
            ui.separator();
            let app_state = self.app_state;
            self.settings.render(ui, &app_state);
        });

        if self.tab == Tab::Chat {
            egui::TopBottomPanel::bottom("chat_input").show(ctx, |ui| {
                self.render_chat_input_panel(ui);
            });
        }

        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            self.render_statusbar(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // 标签页
            ui.horizontal(|ui| {
                for tab in [Tab::Chat, Tab::Prompt, Tab::Model] {
                    let selected = self.tab == tab;
                    ui.selectable_value(&mut self.tab, tab, tab.label());
                    let _ = selected;
                }
            });
            ui.separator();

            let tab = self.tab;
            match tab {
                Tab::Chat => {
                    let (chat, state, target) = (
                        &mut self.chat,
                        self.app_state,
                        self.streaming_target,
                    );
                    chat.render(ui, state, target);
                }
                Tab::Prompt => {
                    let (prompt, settings, state, target) = (
                        &mut self.prompt,
                        &mut self.settings,
                        self.app_state,
                        self.streaming_target,
                    );
                    prompt.render(ui, state, target, settings);
                }
                Tab::Model => {
                    let model = &mut self.model;
                    model.render(ui);
                }
            }
        });
    }
}

impl GgufApp {
    fn render_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("🧠 GGUF 推理引擎");
            ui.separator();

            if ui.button("📂 选择模型").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("GGUF 模型", &["gguf"])
                    .pick_file()
                {
                    self.model_path = path.to_string_lossy().into_owned();
                }
            }

            if !self.model_path.is_empty() {
                ui.monospace(self.model_name.as_str());
                ui.separator();
            }

            if ui.button("加载").clicked() {
                if !self.model_path.is_empty() {
                    self.load_model(self.model_path.clone());
                }
            }

            if ui.button("卸载").clicked() {
                self.reset_chat();
                self.app_state = AppState::Idle;
                self.model_name.clear();
                self.model_summary = None;
                self.model = ModelView::default();
                self.chat = ChatView::default();
                self.prompt = PromptView::default();
                self.error_msg.clear();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                match self.app_state {
                    AppState::Idle => {
                        ui.label("⚪ 未加载模型");
                    }
                    AppState::Loading => {
                        ui.spinner();
                        ui.label("加载中...");
                    }
                    AppState::Ready => {
                        ui.label(
                            egui::RichText::new("🟢 就绪").color(egui::Color32::GREEN),
                        );
                    }
                    AppState::Generating => {
                        ui.spinner();
                        ui.label("🟡 生成中");
                        if ui.button("⏹ 停止").clicked() {
                            self.stop_generation();
                        }
                    }
                    AppState::Error => {
                        ui.label(
                            egui::RichText::new("🔴 错误").color(egui::Color32::RED),
                        );
                    }
                }
            });
        });
        if !self.error_msg.is_empty() {
            ui.label(egui::RichText::new(&self.error_msg).small().color(egui::Color32::RED));
        }
    }

    fn render_statusbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 16.0;
            let s = &self.stats;
            ui.label(format!("ctx: {}/{}", s.ctx_len, s.ctx_limit));
            if s.tok_per_s() > 0.0 {
                ui.label(format!("{:.1} tok/s", s.tok_per_s()));
            }
            if s.elapsed_ms > 0 {
                ui.label(format!("{:.2}s", s.elapsed_ms as f64 / 1000.0));
            }
            if !self.model_name.is_empty() {
                ui.separator();
                ui.label(format!("模型: {}", self.model_name));
            }
        });
    }

    /// 渲染聊天输入面板（独立 BottomPanel，确保焦点正常传递）。
    fn render_chat_input_panel(&mut self, ui: &mut egui::Ui) {
        let ready = self.app_state == AppState::Ready;
        let generating = self.app_state == AppState::Generating
            && self.streaming_target == StreamingTarget::Chat;

        // 上下文超出提示
        if self.chat.ctx_exceeded {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("⚠️ 上下文超出限制，请重置对话")
                        .color(egui::Color32::YELLOW),
                );
                if ui.button("重置").clicked() {
                    self.chat.messages.clear();
                    self.chat.ctx_exceeded = false;
                }
            });
            ui.separator();
        }

        // 多行输入框
        let input_enabled = ready || generating;
        ui.add_enabled_ui(input_enabled, |ui| {
            let mut input = self.chat.input.clone();
            let response = ui.add(
                egui::TextEdit::multiline(&mut input)
                    .desired_width(f32::INFINITY)
                    .desired_rows(3)
                    .hint_text("输入消息... (Enter 发送, Shift+Enter 换行)"),
            );
            self.chat.input = input;

            if response.has_focus()
                && ui.input_mut(|i| i.key_pressed(egui::Key::Enter))
                && !ui.input_mut(|i| i.modifiers.shift)
            {
                let text = self.chat.input.trim().to_string();
                if !text.is_empty() {
                    self.chat.input.clear();
                    self.send_chat(&text);
                }
            }
        });

        // 按钮行
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("发送")).clicked() {
                let text = self.chat.input.trim().to_string();
                if !text.is_empty() {
                    self.chat.input.clear();
                    self.send_chat(&text);
                }
            }
            if ui.add_enabled(ready, egui::Button::new("重置上下文")).clicked() {
                self.reset_chat();
            }
            if generating && ui.button("⏹ 停止").clicked() {
                self.stop_generation();
            }
        });
    }
}
