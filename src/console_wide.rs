//! Windows 宽字符控制台 I/O（UTF-16），绕开代码页转换。
//!
//! PowerShell 管道场景下，`[Console]::WriteLine("中文")` 用控制台代码页
//! （默认 GBK 936）编码后传给子进程 stdin。子进程若按 UTF-8 解析就乱码。
//!
//! 用 `ReadConsoleW` / `WriteConsoleW` 直接读写 UTF-16 宽字符：
//! - **输入**：`ReadConsoleW` 从控制台缓冲区读宽字符，不经过代码页转换。
//!   管道模式下控制台无输入缓冲区，回退到字节级 stdin（UTF-8）。
//! - **输出**：`WriteConsoleW` 写宽字符到控制台，不经过代码页转换。
//!   管道模式下回退到字节级 stdout（UTF-8）。
//!
//! 本模块仅在 `cfg(windows)` 编译，非 Windows 平台为 no-op stub。

#[cfg(windows)]
mod inner {
    use std::io;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, ReadConsoleW, WriteConsoleW, STD_ERROR_HANDLE,
        STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };

    /// 判断一个 raw handle 是否为真实控制台（非管道/文件重定向）。
    fn is_console_raw(raw: HANDLE) -> bool {
        let mut mode: u32 = 0;
        let ret = unsafe { GetConsoleMode(raw, &mut mode) };
        ret != 0
    }

    /// 获取指定标准 handle 并检测是否为控制台。
    ///
    /// `which` 为 `STD_INPUT_HANDLE` / `STD_OUTPUT_HANDLE` / `STD_ERROR_HANDLE`（u32 常量）。
    fn console_check(which: u32) -> bool {
        let h: HANDLE = unsafe { GetStdHandle(which as i32 as u32) };
        // INVALID_HANDLE_VALUE 通常为 0xFFFF_FFFF（-1 as HANDLE）
        if h as i64 == -1 {
            return false;
        }
        is_console_raw(h)
    }

    /// 判断 stdin 是否为真实控制台。
    pub fn stdin_is_console() -> bool {
        console_check(STD_INPUT_HANDLE)
    }

    /// 判断 stdout 是否为真实控制台。
    pub fn stdout_is_console() -> bool {
        console_check(STD_OUTPUT_HANDLE)
    }

    /// 从控制台读取一行宽字符输入。
    ///
    /// 返回 `Ok(Some(String))` 为 UTF-8 文本；`Ok(None)` 表示 EOF。
    /// 管道/重定向场景返回 `Err(NotConnected)`，调用方应回退到字节级 `read_line`。
    pub fn read_line_wide() -> Result<Option<String>, io::Error> {
        let h: HANDLE = unsafe { GetStdHandle(STD_INPUT_HANDLE as i32 as u32) };
        if h as i64 == -1 || !is_console_raw(h) {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "stdin 不是控制台",
            ));
        }
        let mut line: Vec<u16> = Vec::with_capacity(256);
        loop {
            let mut chunk = [0u16; 512];
            let mut n_read: u32 = 0;
            let ok = unsafe {
                ReadConsoleW(
                    h,
                    chunk.as_mut_ptr() as *mut _,
                    chunk.len() as u32,
                    &mut n_read,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            if n_read == 0 {
                if line.is_empty() {
                    return Ok(None);
                }
                break;
            }
            line.extend_from_slice(&chunk[..n_read as usize]);
            // ReadConsoleW 行缓冲：用户按回车后一次性返回整行（含 \r\n）
            if line.iter().any(|&c| c == 0x000A) {
                break;
            }
            if line.len() > 65536 {
                break;
            }
        }
        // 去掉尾部 \r\n / \n
        while line.last() == Some(&0x000A) || line.last() == Some(&0x000D) {
            line.pop();
        }
        Ok(Some(String::from_utf16_lossy(&line)))
    }

    /// 向 stdout 写一段宽字符文本（UTF-8 字符串 → UTF-16）。
    ///
    /// 管道/重定向场景返回 `Err(NotConnected)`，调用方应回退到字节级 `write_all`。
    pub fn write_wide(text: &str) -> Result<(), io::Error> {
        write_to_handle(STD_OUTPUT_HANDLE, text)
    }

    /// 向 stderr 写宽字符（诊断信息用）。
    pub fn write_err_wide(text: &str) -> Result<(), io::Error> {
        write_to_handle(STD_ERROR_HANDLE, text)
    }

    fn write_to_handle(which: u32, text: &str) -> Result<(), io::Error> {
        let h: HANDLE = unsafe { GetStdHandle(which as i32 as u32) };
        if h as i64 == -1 || !is_console_raw(h) {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "不是控制台",
            ));
        }
        let wide: Vec<u16> = text.encode_utf16().collect();
        if wide.is_empty() {
            return Ok(());
        }
        let mut total_written: usize = 0;
        while total_written < wide.len() {
            let remaining = &wide[total_written..];
            let mut n_written: u32 = 0;
            let ok = unsafe {
                WriteConsoleW(
                    h,
                    remaining.as_ptr() as *const _,
                    remaining.len() as u32,
                    &mut n_written,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            if n_written == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "WriteConsoleW 返回 0 字节写入",
                ));
            }
            total_written += n_written as usize;
        }
        Ok(())
    }
}

#[cfg(windows)]
pub use inner::*;

// 非 Windows stub
#[cfg(not(windows))]
pub mod stub {
    use std::io;
    pub fn read_line_wide() -> Result<Option<String>, io::Error> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "not windows"))
    }
    pub fn write_wide(_text: &str) -> Result<(), io::Error> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "not windows"))
    }
    pub fn write_err_wide(_text: &str) -> Result<(), io::Error> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "not windows"))
    }
    pub fn stdin_is_console() -> bool {
        std::io::IsTerminal::is_terminal(&std::io::stdin())
    }
    pub fn stdout_is_console() -> bool {
        std::io::IsTerminal::is_terminal(&std::io::stdout())
    }
}
#[cfg(not(windows))]
pub use stub::*;
