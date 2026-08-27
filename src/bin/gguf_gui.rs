//! GGUF 推理引擎 GUI 入口。
//!
//! 基于 egui (eframe) 的跨平台 GUI，覆盖 gguf-infer 和 gguf-dump 全部命令行功能。
//!
//! 构建: `cargo run --release --features gui --bin gguf-gui`

#[cfg(feature = "gui")]
mod gui {
    #![allow(dead_code)]
    pub mod app;
    pub mod chat_view;
    pub mod inference;
    pub mod model_view;
    pub mod prompt_view;
    pub mod settings;
    pub mod state;
}

#[cfg(feature = "gui")]
fn main() {
    use eframe::egui;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("GGUF 推理引擎"),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "GGUF 推理引擎",
        options,
        Box::new(|cc| {
            // 加载中文字体，解决乱码
            let mut fonts = egui::FontDefinitions::default();
            let font_data = include_bytes!("../../assets/fonts/msyh.ttc");
            fonts
                .font_data
                .insert("msyh".to_owned(), egui::FontData::from_static(font_data));
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.insert(0, "msyh".to_owned());
            }
            cc.egui_ctx.set_fonts(fonts);

            Ok(Box::new(crate::gui::app::GgufApp::new(cc)))
        }),
    ) {
        eprintln!("GUI 启动失败: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("gguf-gui 需要 --features gui 标志");
    eprintln!("构建: cargo run --release --features gui --bin gguf-gui");
    std::process::exit(1);
}
