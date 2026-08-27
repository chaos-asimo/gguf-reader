//! 模型信息视图。

use eframe::egui;

use super::state::ModelSummary;

const TENSOR_LIMIT: usize = 50;

/// 模型信息视图状态
#[derive(Default)]
pub struct ModelView {
    pub summary: Option<ModelSummary>,
    pub kv_data: Vec<(String, String, String)>,
    pub tensor_data: Vec<(String, String, String)>,
    pub kv_search: String,
    pub show_all_tensors: bool,
    pub token_id_query: String,
    pub token_str_query: String,
    pub token_result: String,
    pub ctx_limit: usize,
    pub embed_dim: u32,
    pub vocab_size: u32,
}

impl ModelView {
    pub fn render(&mut self, ui: &mut egui::Ui) {
        if self.summary.is_none() {
            ui.centered_and_justified(|ui| {
                ui.add_space(40.0);
                ui.label(
                    egui::RichText::new("请先加载模型")
                        .color(egui::Color32::GRAY),
                );
            });
            return;
        }

        let summary = self.summary.clone().unwrap();

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                // 文件摘要
                ui.group(|ui| {
                    ui.label(egui::RichText::new("文件摘要").strong());
                    ui.label(format!("文件名: {}", summary.name));
                    ui.label(format!("架构: {}", summary.arch));
                    ui.label(format!("模型名: {}", summary.model_name));
                    ui.label(format!("GGUF 版本: {}", summary.gguf_version));
                    ui.label(format!("对齐: {}", summary.alignment));
                    ui.label(format!("数据偏移: {}", summary.data_offset));
                    ui.label(format!("文件大小: {:.1} MB", summary.size_mb()));
                    ui.label(format!("张量数: {}", summary.tensor_count));
                    ui.label(format!("KV 数: {}", summary.kv_count));
                    ui.label(format!("加载耗时: {} ms", summary.load_ms));
                    if self.vocab_size > 0 {
                        ui.label(format!("词表大小: {}", self.vocab_size));
                    }
                    if self.embed_dim > 0 {
                        ui.label(format!("隐藏维度: {}", self.embed_dim));
                    }
                    if self.ctx_limit > 0 {
                        ui.label(format!("上下文长度: {}", self.ctx_limit));
                    }
                });
                ui.add_space(8.0);

                // KV 元数据
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("KV 元数据").strong());
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label("搜索:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.kv_search)
                                        .hint_text("过滤键名...")
                                        .desired_width(160.0),
                                );
                            },
                        );
                    });
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "{:<50} {:<10} {}",
                            "键", "类型", "值"
                        ))
                        .monospace()
                        .strong(),
                    );
                    ui.separator();

                    let filtered: Vec<&(String, String, String)> =
                        if self.kv_search.is_empty() {
                            self.kv_data.iter().collect()
                        } else {
                            let q = self.kv_search.to_lowercase();
                            self.kv_data
                                .iter()
                                .filter(|(k, _, _)| k.to_lowercase().contains(&q))
                                .collect()
                        };

                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for (k, t, v) in &filtered {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{:<50} {:<10} {}",
                                        k, t, v
                                    ))
                                    .monospace()
                                    .small(),
                                );
                            }
                            if filtered.is_empty() {
                                ui.label(
                                    egui::RichText::new("(无匹配项)")
                                        .color(egui::Color32::GRAY),
                                );
                            }
                        });
                });
                ui.add_space(8.0);

                // 张量列表
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("张量列表").strong());
                        let label = if self.show_all_tensors {
                            "显示全部"
                        } else {
                            &format!("显示全部 (当前前 {TENSOR_LIMIT})")
                        };
                        ui.checkbox(&mut self.show_all_tensors, label);
                    });
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "{:<50} {:<25} {}",
                            "名称", "形状", "类型"
                        ))
                        .monospace()
                        .strong(),
                    );
                    ui.separator();

                    let limit = if self.show_all_tensors {
                        self.tensor_data.len()
                    } else {
                        self.tensor_data.len().min(TENSOR_LIMIT)
                    };

                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for (name, shape, dtype) in self.tensor_data.iter().take(limit) {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{:<50} {:<25} {}",
                                        name, shape, dtype
                                    ))
                                    .monospace()
                                    .small(),
                                );
                            }
                            if self.tensor_data.len() > limit {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "... 还有 {} 个张量 (勾选\"显示全部\"查看)",
                                        self.tensor_data.len() - limit
                                    ))
                                    .color(egui::Color32::GRAY)
                                    .small(),
                                );
                            }
                        });
                });
                ui.add_space(8.0);

                // 词表查询
                ui.group(|ui| {
                    ui.label(egui::RichText::new("词表查询").strong());
                    ui.horizontal(|ui| {
                        ui.label("Token ID:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.token_id_query)
                                .hint_text("如 151645")
                                .desired_width(100.0),
                        );
                        ui.label("Token 字符串:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.token_str_query)
                                .hint_text("如 im_end")
                                .desired_width(120.0),
                        );
                        if ui.button("查询").clicked() {
                            self.token_result = format!(
                                "ID=\"{}\" / STR=\"{}\" — 需在模型加载后通过 tokenizer 查询",
                                self.token_id_query, self.token_str_query
                            );
                        }
                    });
                    if !self.token_result.is_empty() {
                        ui.label(egui::RichText::new(&self.token_result).small());
                    }
                });
                ui.add_space(8.0);

                // 导出
                ui.horizontal(|ui| {
                    if ui.button("📄 导出 JSON").clicked() {
                        let json = self.export_json();
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name("model_info.json")
                            .add_filter("JSON", &["json"])
                            .save_file()
                        {
                            let _ = std::fs::write(path, json);
                        }
                    }
                    if ui.button("📋 显示摘要").clicked() {
                        self.token_result = self.summary_text();
                    }
                });
            });
    }

    fn export_json(&self) -> String {
        let Some(summary) = &self.summary else {
            return "{}".into();
        };
        let kv = self
            .kv_data
            .iter()
            .map(|(k, t, v)| format!("    {{\"key\": {:?}, \"type\": {:?}, \"value\": {:?}}}", k, t, v))
            .collect::<Vec<_>>()
            .join(",\n");
        let tensors = self
            .tensor_data
            .iter()
            .map(|(n, s, d)| format!("    {{\"name\": {:?}, \"shape\": {:?}, \"dtype\": {:?}}}", n, s, d))
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            "{{\n  \"summary\": {{\n    \"name\": {:?},\n    \"arch\": {:?},\n    \"model_name\": {:?},\n    \"gguf_version\": {},\n    \"alignment\": {},\n    \"data_offset\": {},\n    \"file_size\": {},\n    \"tensor_count\": {},\n    \"kv_count\": {},\n    \"load_ms\": {}\n  }},\n  \"kv\": [\n{}\n  ],\n  \"tensors\": [\n{}\n  ]\n}}",
            summary.name,
            summary.arch,
            summary.model_name,
            summary.gguf_version,
            summary.alignment,
            summary.data_offset,
            summary.file_size,
            summary.tensor_count,
            summary.kv_count,
            summary.load_ms,
            kv,
            tensors
        )
    }

    fn summary_text(&self) -> String {
        let Some(s) = &self.summary else {
            return String::new();
        };
        format!(
            "模型: {}\n架构: {}\nGGUF 版本: {}\n对齐: {}\n数据偏移: {}\n文件大小: {:.1} MB\n张量数: {}\nKV 数: {}\n加载耗时: {} ms",
            s.name,
            s.arch,
            s.gguf_version,
            s.alignment,
            s.data_offset,
            s.size_mb(),
            s.tensor_count,
            s.kv_count,
            s.load_ms
        )
    }
}
