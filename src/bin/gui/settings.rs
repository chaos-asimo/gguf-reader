//! 采样参数设置面板。

use eframe::egui;

use super::inference::{UiCommand};
use super::state::{AppState, GenStats};

/// 设置面板
#[derive(Clone)]
pub struct SettingsPanel {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub greedy: bool,
    pub collapsed: bool,
    /// 命令发送端（用于发送对话/prompt 命令）
    pub(crate) cmd_tx: Option<std::sync::mpsc::Sender<UiCommand>>,
    /// 停止标志
    pub(crate) stop_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// 生成统计（供 prompt_view 显示）
    pub stats: GenStats,
}

impl Default for SettingsPanel {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.1,
            seed: 0,
            max_tokens: 512,
            greedy: false,
            collapsed: true,
            cmd_tx: None,
            stop_flag: None,
            stats: GenStats::default(),
        }
    }
}

impl SettingsPanel {
    /// 构建当前采样配置。
    pub fn sampler_config(&self) -> gguf::infer::sampler::SamplerConfig {
        if self.greedy {
            gguf::infer::sampler::SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                seed: self.seed,
            }
        } else {
            gguf::infer::sampler::SamplerConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                min_p: self.min_p,
                repeat_penalty: self.repeat_penalty,
                seed: self.seed,
            }
        }
    }

    /// 发送对话命令到推理线程。
    pub fn send_chat(&self, text: String) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(UiCommand::Chat {
                text,
                max_tokens: self.max_tokens,
                sampler: self.sampler_config(),
            });
        }
    }

    /// 停止当前生成。
    pub fn stop_generation(&self) {
        if let Some(flag) = &self.stop_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// 发送 prompt 补全命令到推理线程。
    pub fn send_prompt(&self, text: String, stream: bool) {
        if text.is_empty() {
            return;
        }
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(UiCommand::Prompt {
                text,
                max_tokens: self.max_tokens,
                sampler: self.sampler_config(),
                stream,
            });
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui, app_state: &AppState) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚙ 设置").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.toggle_value(&mut self.collapsed, "折叠");
            });
        });

        if self.collapsed {
            return;
        }

        ui.horizontal_wrapped(|ui| {
            let disabled = *app_state == AppState::Generating;
            let greedy_disabled = self.greedy || disabled;

            ui.add_enabled_ui(!greedy_disabled, |ui| {
                ui.label("温度:");
                ui.add(
                    egui::Slider::new(&mut self.temperature, 0.0..=2.0)
                        .text("温度")
                        .fixed_decimals(2)
                        .suffix(" "),
                );
            });

            ui.add_enabled_ui(!greedy_disabled, |ui| {
                ui.label("Top-K:");
                ui.add(egui::DragValue::new(&mut self.top_k).range(0..=1000));
            });

            ui.add_enabled_ui(!greedy_disabled, |ui| {
                ui.label("Top-P:");
                ui.add(
                    egui::Slider::new(&mut self.top_p, 0.0..=1.0)
                        .fixed_decimals(2)
                        .suffix(" "),
                );
            });

            ui.add_enabled_ui(!greedy_disabled, |ui| {
                ui.label("Min-P:");
                ui.add(
                    egui::Slider::new(&mut self.min_p, 0.0..=1.0)
                        .fixed_decimals(2)
                        .suffix(" "),
                );
            });

            ui.add_enabled_ui(!greedy_disabled, |ui| {
                ui.label("重复惩罚:");
                ui.add(
                    egui::Slider::new(&mut self.repeat_penalty, 1.0..=2.0)
                        .fixed_decimals(2)
                        .suffix(" "),
                );
            });

            ui.add_enabled_ui(!disabled, |ui| {
                ui.label("种子:");
                ui.add(egui::DragValue::new(&mut self.seed));
            });

            ui.add_enabled_ui(!disabled, |ui| {
                ui.label("最大Token:");
                ui.add(egui::DragValue::new(&mut self.max_tokens).range(1..=32768));
            });

            ui.add_enabled_ui(!disabled, |ui| {
                ui.checkbox(&mut self.greedy, "贪心");
            });
        });
    }
}
