//! Prompt 补全视图。

use eframe::egui;

use super::state::AppState;
use super::app::StreamingTarget;
use super::settings::SettingsPanel;

/// Prompt 补全视图状态
#[derive(Default)]
pub struct PromptView {
    pub input_text: String,
    pub output: String,
    pub stream: bool,
    pub greedy: bool,
}

impl PromptView {
    pub fn render(
        &mut self,
        ui: &mut egui::Ui,
        app_state: AppState,
        streaming_target: StreamingTarget,
        settings: &mut SettingsPanel,
    ) {
        let ready = app_state == AppState::Ready;
        let generating = app_state == AppState::Generating
            && streaming_target == StreamingTarget::Prompt;

        // Prompt 输入区
        ui.label("Prompt 输入:");
        let mut input = self.input_text.clone();
        ui.add(
            egui::TextEdit::multiline(&mut input)
                .desired_width(f32::INFINITY)
                .desired_rows(4)
                .hint_text("输入 prompt..."),
        );
        self.input_text = input;

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.stream, "流式输出");
            ui.checkbox(&mut self.greedy, "贪心解码");
        });

        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("▶ 运行")).clicked() {
                settings.greedy = self.greedy;
                settings.send_prompt(self.input_text.trim().to_string(), self.stream);
            }
            if ui.button("清除输出").clicked() {
                self.output.clear();
            }
            if generating && ui.button("⏹ 停止").clicked() {
                settings.stop_generation();
            }
        });

        ui.separator();

        // 输出区
        ui.label("输出:");
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.output.is_empty() {
                    ui.label(
                        egui::RichText::new("运行后显示输出...")
                            .color(egui::Color32::GRAY),
                    );
                } else {
                    ui.label(&self.output);
                    if generating {
                        ui.label(
                            egui::RichText::new("● 生成中...")
                                .small()
                                .color(egui::Color32::YELLOW),
                        );
                    }
                }
            });

        // 统计
        let s = &settings.stats;
        if s.elapsed_ms > 0 {
            ui.horizontal(|ui| {
                ui.weak(format!(
                    "耗时 {:.2}s | {:.1} tok/s | ctx {}/{}",
                    s.elapsed_ms as f64 / 1000.0,
                    s.tok_per_s(),
                    s.ctx_len,
                    s.ctx_limit
                ));
            });
        }
    }
}
