//! 对话视图。

use eframe::egui;

use super::state::{AppState, ChatMessage, Role};
use super::app::StreamingTarget;

/// 对话视图状态
#[derive(Default)]
pub struct ChatView {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub ctx_exceeded: bool,
}

impl ChatView {
    pub fn render(
        &mut self,
        ui: &mut egui::Ui,
        app_state: AppState,
        streaming_target: StreamingTarget,
    ) {
        let generating = app_state == AppState::Generating
            && streaming_target == StreamingTarget::Chat;

        // 上下文超出提示
        if self.ctx_exceeded {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("⚠️ 上下文超出限制，请重置对话")
                        .color(egui::Color32::YELLOW),
                );
                if ui.button("重置").clicked() {
                    self.reset();
                }
            });
            ui.separator();
        }

        // 对话区：填满全部可用空间（输入框由 app.rs 的 BottomPanel 独立渲染）
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.messages.is_empty() {
                    ui.add_space(ui.available_height() * 0.3);
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new("输入消息开始对话")
                                .color(egui::Color32::GRAY),
                        );
                    });
                    return;
                }

                for msg in &self.messages {
                    let is_user = msg.role == Role::User;
                    let bg = if is_user {
                        egui::Color32::from_rgb(70, 130, 180)
                    } else {
                        egui::Color32::from_rgb(60, 60, 60)
                    };
                    let text_color = egui::Color32::WHITE;

                    ui.horizontal_wrapped(|ui| {
                        egui::Frame::default()
                            .fill(bg)
                            .inner_margin(egui::Margin::symmetric(10.0, 6.0))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(msg.role.label())
                                        .small()
                                        .strong()
                                        .color(text_color),
                                );
                                ui.label(
                                    egui::RichText::new(&msg.content)
                                        .color(text_color),
                                );
                            });
                    });
                    ui.add_space(6.0);
                }

                if generating {
                    ui.label(
                        egui::RichText::new("● 生成中...")
                            .small()
                            .color(egui::Color32::YELLOW),
                    );
                }
            });
    }

    fn reset(&mut self) {
        self.messages.clear();
        self.ctx_exceeded = false;
    }
}
