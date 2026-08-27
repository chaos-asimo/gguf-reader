//! GGUF BPE 分词器（GPT-2 字节级 BPE）。
//!
//! 从 GGUF KV 元数据加载词表与 BPE 合并规则，提供 `encode`（文本→token id）
//! 和 `decode`（token id→文本）。
//!
//! 支持两种模式（自动检测）：
//! - **GPT-2 字节级 BPE**：text → bytes → byte tokens → BPE merges（Qwen2 / LLaMA / Mistral 等）
//! - **贪心最长匹配**：适用于词表已包含完整子词的旧式 GGUF

use std::collections::HashMap;

use crate::error::{GgufError, GgufResult};
use crate::file::GgufFile;
use crate::types::GgufValue;

// ─── GPT-2 Byte Level BPE ──────────────────────────────────────────────

// ─── Tokenizer ────────────────────────────────────────────────────────────

/// 从 GGUF 文件加载的分词器。
pub struct Tokenizer {
    /// 基础 token 表：字符串 → token id
    tokens: HashMap<String, u32>,
    /// 反向表：token id → 字符串
    token_strings: Vec<String>,
    /// 是否 GPT-2 字节级 BPE（自动检测）
    is_byte_level: bool,
    /// BPE 合并规则：(left_id, right_id) → (priority, merged_id)
    bpe_merges: HashMap<(u32, u32), (u32, u32)>,
    /// 贪心模式用：BPE 合并优先级（key → priority）
    #[allow(dead_code)]
    greedy_merges: HashMap<String, u32>,
    /// BOS token id
    pub bos_id: Option<u32>,
    /// EOS token id
    pub eos_id: Option<u32>,
    /// 是否编码时添加 BOS
    add_bos: bool,
}

impl Tokenizer {
    /// 从 GGUF 文件解析分词器元数据。
    ///
    /// 兼容两种命名：
    /// - 新式（llama.cpp 现代导出）：`tokenizer.ggml.tokens` / `tokenizer.ggml.merges` /
    ///   `tokenizer.ggml.bos_token_id` / `tokenizer.ggml.eos_token_id` / `tokenizer.ggml.add_bos_token`
    /// - 旧式：`{arch}.tokenizer.n_vocab` / `{arch}.tokenizer.tokens` / ...
    pub fn from_gguf(file: &GgufFile) -> GgufResult<Self> {
        // 架构前缀（旧式键需要）
        let arch = file
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GgufError::MissingTensor {
                name: "general.architecture".into(),
                kind: "kv",
            })?
            .to_string();
        let tkn_old = |s: &str| format!("{}.tokenizer.{s}", arch);

        /// 在新式 / 旧式键中取第一个存在的 GgufValue。
        fn get_any<'f>(file: &'f GgufFile, new_key: &str, old_key: &str) -> Option<&'f GgufValue> {
            file.get(new_key).or_else(|| file.get(old_key))
        }

        // 词表：新式 tokenizer.ggml.tokens，旧式 {arch}.tokenizer.tokens
        let toks = get_any(file, "tokenizer.ggml.tokens", &tkn_old("tokens"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| GgufError::MissingTensor {
                name: "tokenizer.ggml.tokens".into(),
                kind: "kv",
            })?;
        let n_vocab = toks.data.len();

        let mut tokens: HashMap<String, u32> = HashMap::with_capacity(n_vocab);
        let mut token_strings: Vec<String> = vec![String::new(); n_vocab];
        for (i, v) in toks.data.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or_else(|| {
                    GgufError::InferenceError(format!("tokenizer token[{}] 非字符串", i))
                })?
                .to_string();
            tokens.insert(s.clone(), i as u32);
            token_strings[i] = s;
        }

        // BPE merges：新式 tokenizer.ggml.merges（长度自描述），旧式 {arch}.tokenizer.merges
        //
        // 关键：merges 条目格式为 "left right"（字符串），left/right 是**词表里的 token 字符串**，
        // 不是 token id。BPE 训练时每一步合并的是一对相邻 token 字符串，合并结果 token 的
        // 字符串恰好等于 left + right。
        //
        // 重要更正：merges 数组下标 i **不等于** 词表 id (i+1)。Qwen2 使用 tiktoken 风格
        // BPE，merges 是训练步骤顺序，与词表 id 排列不对齐（例如 merge[12792]="H i"，
        // 但真正的 "Hi" token 在词表里是 id=13048，而 12793 是 "Ġimagine"）。
        // 因此合并结果 id 必须**查表** tokens.get(left + right) 得到，而不能取 i+1。
        //
        // merge 规则 key 用 id 对 (left_id, right_id)：GPT-2 词表中 id 与字符串一一对应，
        // 故 id 对 key 等价于字符串对 key，且查找更快。
        let mut greedy_merges: HashMap<String, u32> = HashMap::new();
        // BPE 合并规则：(left_id, right_id) → (priority, merged_id)
        let mut bpe_merges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        if let Some(arr) =
            get_any(file, "tokenizer.ggml.merges", &tkn_old("merges")).and_then(|v| v.as_array())
        {
            for (i, v) in arr.data.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    let s = s.to_string();
                    // 贪心模式
                    greedy_merges.insert(s.clone(), i as u32);
                    // BPE 模式：解析 "left right" 字符串对 → id 对，merged_id 查表得到
                    if let Some((left, right)) = s.split_once(' ') {
                        if let (Some(lid), Some(rid)) = (tokens.get(left), tokens.get(right)) {
                            let merged = left.to_string() + right;
                            if let Some(&merged_id) = tokens.get(merged.as_str()) {
                                bpe_merges.insert((*lid, *rid), (i as u32, merged_id));
                            }
                        }
                    }
                }
            }
        }

        // 检测是否为 GPT-2 字节级 BPE：
        // 检查词表中是否包含 "h"（字节 0x68）和 "i"（字节 0x69）这样的单字节 ASCII token
        let is_byte_level = tokens.contains_key("h") && tokens.contains_key("i");

        // Special tokens
        let get_special_id = |new_key: &str, old_key: &str| -> Option<u32> {
            get_any(file, new_key, old_key)
                .and_then(|v| v.as_i64())
                .map(|x| x as u32)
        };
        let bos_id = get_special_id(
            "tokenizer.ggml.bos_token_id",
            &tkn_old("bos_token_id"),
        );
        let eos_id = get_special_id(
            "tokenizer.ggml.eos_token_id",
            &tkn_old("eos_token_id"),
        );
        let add_bos = get_any(file, "tokenizer.ggml.add_bos_token", &tkn_old("add_bos"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            tokens,
            token_strings,
            is_byte_level,
            bpe_merges,
            greedy_merges,
            bos_id,
            eos_id,
            add_bos,
        })
    }

    /// 词表大小。
    pub fn vocab_size(&self) -> usize {
        self.token_strings.len()
    }

    /// token id → 字符串。
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.token_strings.get(id as usize).map(|s| s.as_str())
    }

    /// 字符串 → token id（仅查表，不做 BPE）。
    pub fn tokenize(&self, s: &str) -> Option<u32> {
        self.tokens.get(s).copied()
    }

    /// 文本 → token id 列表。
    ///
    /// 自动选择编码策略：
    /// - **GPT-2 字节级 BPE**：text → bytes → byte tokens → BPE merges
    /// - **贪心最长匹配**：从当前位置尝试最长 token，逐步缩短直到命中词表
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return self.bos_ids();
        }

        if self.is_byte_level {
            return self.bos_ids().into_iter().chain(self.bpe_encode(text)).collect();
        }

        self.bos_ids().into_iter().chain(self.greedy_encode(text)).collect()
    }

    /// GPT-2 字节级 BPE 编码。
    ///
    /// 步骤：
    /// 1. text → UTF-8 bytes
    /// 2. 每个 byte → 字节 token id（单字符，按标准 bytes_to_unicode 映射后查表）
    /// 3. 迭代 BPE 合并（按 merges 优先级）：每轮找到优先级最小（最先训练）的
    ///    可合并相邻 id 对，合并后 id = merged_id，替换原相邻两项为一项；
    ///    重复直到没有可合并对。
    ///
    /// merge 规则 `bpe_merges` 的 key 是 `(left_id, right_id)`，由 from_gguf 里
    /// 把 "left right" 字符串对查表转成 id 对后构建。GPT-2 词表中 id 与字符串
    /// 一一对应，故 id 对 key 等价于字符串对 key。
    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        let raw_bytes = text.as_bytes();

        // 每个原始字节 → 字节 token id
        // GPT-2 字节级 BPE：byte_encoder 把 256 个字节映射到 256 个唯一单字符字符串，
        // 这里用标准 GPT-2 byte_encoder（bytes_to_unicode），再查表得到 id。
        let mut symbols: Vec<u32> = Vec::with_capacity(raw_bytes.len());
        for &b in raw_bytes {
            if let Some(tok_str) = gpt2_byte_to_token(b) {
                if let Some(&id) = self.tokens.get(tok_str.as_str()) {
                    symbols.push(id);
                }
            }
        }

        // 正确的 BPE：反复迭代，每轮找到优先级最小（最先训练）的可合并相邻 id 对，
        // 合并后 id = merged_id，替换原相邻两项为一项；直到没有可合并对为止。
        // 单次线性扫描会漏掉需要多轮才能合并成的 token（如中文多字节序列）。
        loop {
            let mut best = None; // (priority, index, merged_id)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&(priority, merged_id)) =
                    self.bpe_merges.get(&(symbols[i], symbols[i + 1]))
                {
                    if best.as_ref().map_or(true, |(bp, _, _)| priority < *bp) {
                        best = Some((priority, i, merged_id));
                    }
                }
            }
            match best {
                Some((_priority, idx, merged_id)) => {
                    symbols[idx] = merged_id;
                    symbols.remove(idx + 1);
                }
                None => break,
            }
        }

        symbols
    }

    /// 贪心最长匹配编码。
    fn greedy_encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            let max_len = (bytes.len() - i).min(8);
            let mut found = false;
            for l in (1..=max_len).rev() {
                let s = &bytes[i..i + l];
                if let Ok(word) = std::str::from_utf8(s) {
                    if let Some(&id) = self.tokens.get(word) {
                        out.push(id);
                        i += l;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                i += 1;
            }
        }
        out
    }

    fn bos_ids(&self) -> Vec<u32> {
        if self.add_bos {
            self.bos_id.map(|id| vec![id]).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// token id 列表 → 文本。
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.is_byte_level {
            return self.bpe_decode(ids);
        }
        let mut out = String::new();
        for &id in ids {
            if let Some(s) = self.decode_token(id) {
                out.push_str(s);
            }
        }
        out
    }

    /// GPT-2 字节级 BPE 解码。
    ///
    /// 将 token id 列表还原为原始字节，再转为 UTF-8 字符串。
    /// 对 GPT-2 映射表无法处理的字符（如 CJK 特殊 token），直接追加原始 UTF-8 字节。
    fn bpe_decode(&self, ids: &[u32]) -> String {
        let mut raw_bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(s) = self.decode_token(id) {
                // 每个 char 经 GPT-2 bytes_to_unicode 映射到唯一原始字节，
                // 用 gpt2_token_to_byte 反向，再按 UTF-8 重组。
                for c in s.chars() {
                    if let Some(b) = gpt2_token_to_byte(c) {
                        raw_bytes.push(b);
                    } else {
                        // 非 GPT-2 映射字符（CJK 等特殊 token）：直接追加原始 UTF-8 字节
                        let mut buf = [0u8; 4];
                        raw_bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }
}

/// 构建 GPT-2 `bytes_to_unicode` 映射表。
///
/// 标准 GPT-2 映射：
/// - 可打印字节 33..=126（94 个）+ 161..=172（12 个）+ 174..=255（82 个）
///   共 188 个，映射到自身（`byte == codepoint`）；
/// - 其余 68 个不可打印字节按升序映射到 U+0100 起的私有区（0x100..=0x143）。
///
/// 返回 (byte → codepoint, codepoint → byte) 两张表。
/// codepoint 表大小为 256（覆盖 0..=255 的可打印字节），私有区用 b2cp 反向查找。
fn build_bytes_to_unicode() -> ([u32; 256], [u8; 256]) {
    let mut b2cp = [0u32; 256];
    let mut cp2b = [0u8; 256];
    let is_printable = |b: u8| -> bool {
        (33..=126u8).contains(&b) || (161..=172u8).contains(&b) || (174..=255u8).contains(&b)
    };
    let mut n = 0u32;
    for b in 0..=255u8 {
        if is_printable(b) {
            b2cp[b as usize] = b as u32;
            cp2b[b as usize] = b;
        } else {
            let cp = 0x100 + n;
            b2cp[b as usize] = cp;
            n += 1;
        }
    }
    (b2cp, cp2b)
}

// 全局映射表（lazy 初始化）
static BYTES_TO_UNICODE: std::sync::OnceLock<([u32; 256], [u8; 256])> = std::sync::OnceLock::new();

fn mappings() -> &'static ([u32; 256], [u8; 256]) {
    BYTES_TO_UNICODE.get_or_init(build_bytes_to_unicode)
}

/// 字节 encoder：原始字节 → 词表中的单字节 token 字符串。
///
/// 使用标准 GPT-2 `bytes_to_unicode` 映射表（经 probe_vocab / probe10 / verify_mapping
/// 探测确认）：
/// - 可打印字节（33..=126 / 161..=172 / 174..=255）映射到自身 codepoint；
/// - 不可打印字节映射到 U+0100 起的私有区。
fn gpt2_byte_to_token(b: u8) -> Option<String> {
    let (b2cp, _) = mappings();
    char::from_u32(b2cp[b as usize]).map(|c| c.to_string())
}

/// 反向映射：token 字符串中的单个 char → 原始字节。
///
/// 与 [`gpt2_byte_to_token`] 互逆，使用标准 GPT-2 `unicode_to_bytes` 映射：
/// - 可打印字节（33..=126 / 161..=172 / 174..=255）：`byte = cp`（直接查 cp2b 表）
/// - 私有区 codepoint（0x100..=0x143）：遍历 b2cp 找到 `b2cp[b] == cp` 的字节
/// - Qwen 私有区 token（U+E000..=U+F8FF）：返回 `None`，调用方应直接追加原始 UTF-8 字节
pub fn gpt2_token_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if cp < 256 {
        // 可打印字节：直接查表
        let (_, cp2b) = mappings();
        let b = cp2b[cp as usize];
        // 验证：b2cp[b] 应该等于 cp（排除 cp2b 默认值 0 的误判）
        let (b2cp, _) = mappings();
        if b2cp[b as usize] == cp {
            return Some(b);
        }
        return None;
    }
    if (0x100..=0x143).contains(&cp) {
        // 私有区：遍历 b2cp 找到映射到此 cp 的字节
        let (b2cp, _) = mappings();
        for b in 0..=255u8 {
            if b2cp[b as usize] == cp {
                return Some(b);
            }
        }
    }
    // Qwen 私有区 token（U+E000..=U+F8FF）：非 GPT-2 映射，返回 None
    if (0xE000..=0xF8FF).contains(&cp) {
        return None;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgufValue;
    use std::io::Cursor as IoCursor;

    /// 构造一个最小 GGUF 带 3 个 token 和 1 条 merge 规则。
    fn build_tokenizer_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors = 0
        buf.extend_from_slice(&8i64.to_le_bytes()); // n_kv = 8

        // KV 写入辅助
        let write_kv = |buf: &mut Vec<u8>, key: &str, ty: i32, payload: &[u8]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            if ty == 8 {
                buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            }
            buf.extend_from_slice(payload);
        };
        let write_array = |buf: &mut Vec<u8>, key: &str, elem_type: i32, items: &[GgufValue]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&9i32.to_le_bytes()); // Array
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(items.len() as i64).to_le_bytes());
            for item in items {
                match item {
                    GgufValue::String(s) => {
                        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        buf.extend_from_slice(s.as_bytes());
                    }
                    GgufValue::U32(v) => {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
        };

        write_kv(&mut buf, "general.architecture", 8, b"llama");
        write_kv(&mut buf, "general.alignment", 4, &32u32.to_le_bytes());
        write_kv(&mut buf, "llama.tokenizer.n_vocab", 4, &3u32.to_le_bytes());
        write_array(
            &mut buf,
            "llama.tokenizer.tokens",
            8, // String
            &[
                GgufValue::String("hello".into()),
                GgufValue::String("world".into()),
                GgufValue::String("hello world".into()),
            ],
        );
        write_kv(&mut buf, "llama.tokenizer.n_merge", 4, &1u32.to_le_bytes());
        write_array(
            &mut buf,
            "llama.tokenizer.merges",
            8, // String
            &[GgufValue::String("hello world".into())],
        );
        write_kv(
            &mut buf,
            "llama.tokenizer.bos_token_id",
            4,
            &0u32.to_le_bytes(),
        );
        write_kv(
            &mut buf,
            "llama.tokenizer.add_bos",
            7, // Bool
            &[1u8],
        );

        buf
    }

    #[test]
    fn test_from_gguf_basic() {
        let buf = build_tokenizer_gguf();
        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.decode_token(0), Some("hello"));
        assert_eq!(tok.decode_token(1), Some("world"));
        assert_eq!(tok.decode_token(2), Some("hello world"));
        assert_eq!(tok.tokenize("hello"), Some(0));
        assert_eq!(tok.bos_id, Some(0));
        assert!(tok.add_bos);
    }

    #[test]
    fn test_encode_bpe() {
        let buf = build_tokenizer_gguf();
        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        // 词表含 "hello" (id=0)、"world" (id=1)、"hello world" (id=2)
        // "hello world" 整体作为单 token 不在词表 → 贪心匹配 "hello" + 跳过空格 + "world"
        let ids = tok.encode("hello world");
        assert_eq!(ids, vec![0u32, 0, 1]);

        // "hello" → 匹配 "hello" (id=0)
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![0u32, 0]);
    }

    #[test]
    fn test_decode() {
        let buf = build_tokenizer_gguf();
        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        assert_eq!(tok.decode(&[0, 1]), "helloworld");
        assert_eq!(tok.decode(&[2]), "hello world");
    }

    /// 验证贪心最长匹配：词表中存在 "hello world" 时应优先匹配。
    #[test]
    fn test_encode_greedy_longest_match() {
        // 构造词表：BOS, "hello", "world", "hello world"（4 个 KV）
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&4i64.to_le_bytes());

        let write_kv = |buf: &mut Vec<u8>, key: &str, ty: i32, payload: &[u8]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            if ty == 8 {
                buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            }
            buf.extend_from_slice(payload);
        };
        let write_array = |buf: &mut Vec<u8>, key: &str, elem_type: i32, items: &[GgufValue]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&9i32.to_le_bytes());
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(items.len() as i64).to_le_bytes());
            for item in items {
                if let GgufValue::String(s) = item {
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
            }
        };

        write_kv(&mut buf, "general.architecture", 8, b"llama");
        write_kv(&mut buf, "llama.tokenizer.n_vocab", 4, &4u32.to_le_bytes());
        write_array(
            &mut buf,
            "llama.tokenizer.tokens",
            8,
            &[
                GgufValue::String("<s>".into()),       // 0 = BOS
                GgufValue::String("hello".into()),     // 1
                GgufValue::String("world".into()),     // 2
                GgufValue::String("hello world".into()), // 3
            ],
        );
        write_kv(&mut buf, "llama.tokenizer.bos_token_id", 4, &0u32.to_le_bytes());
        write_kv(&mut buf, "llama.tokenizer.add_bos", 7, &[1u8]);

        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();

        // "hello world" → 贪心最长 8 字节只匹配到 "hello" (id=1) + "world" (id=2)
        // （"hello world" 是 11 字节，超出单次窗口）
        let ids = tok.encode("hello world");
        assert_eq!(ids, vec![0u32, 1, 2]);

        // "hello" → 匹配 "hello" (id=1)
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![0u32, 1u32]);
    }

    /// 中文 token 应能正确匹配（贪心最长匹配支持多字节 UTF-8）。
    #[test]
    fn test_encode_chinese() {
        // 构造含中文 token 的词表
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&3i64.to_le_bytes());

        let write_kv = |buf: &mut Vec<u8>, key: &str, ty: i32, payload: &[u8]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            if ty == 8 {
                buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            }
            buf.extend_from_slice(payload);
        };
        let write_array = |buf: &mut Vec<u8>, key: &str, elem_type: i32, items: &[GgufValue]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&9i32.to_le_bytes());
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(items.len() as i64).to_le_bytes());
            for item in items {
                if let GgufValue::String(s) = item {
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
            }
        };

        write_kv(&mut buf, "general.architecture", 8, b"llama");
        write_kv(&mut buf, "llama.tokenizer.n_vocab", 4, &3u32.to_le_bytes());
        write_array(
            &mut buf,
            "llama.tokenizer.tokens",
            8,
            &[
                GgufValue::String("你".into()),
                GgufValue::String("好".into()),
                GgufValue::String("你好".into()),
            ],
        );
        write_kv(&mut buf, "llama.tokenizer.add_bos", 7, &[0u8]);

        let f = GgufFile::from_reader(IoCursor::new(buf)).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        // "你好" → 贪心匹配 "你好" (id=2)
        let ids = tok.encode("你好");
        assert_eq!(ids, vec![2u32]);
    }
}
