//! LLM 推理引擎。
//!
//! 组合 [`GgufFile`] + [`LlamaModel`] + [`Tokenizer`] + [`Sampler`]，
//! 提供 `generate`（逐 token 生成）和 `complete`（一次性补全）接口。

use crate::error::{GgufError, GgufResult};
use crate::file::GgufFile;
use crate::infer::model::llama::LlamaModel;
use crate::infer::sampler::{Sampler, SamplerConfig};
use crate::infer::tokenizer::{gpt2_token_to_byte, Tokenizer};

/// 推理引擎：加载 GGUF 模型文件，执行文本生成。
pub struct Engine<'a> {
    #[allow(dead_code)]
    file: &'a GgufFile,
    model: LlamaModel<'a>,
    tokenizer: Tokenizer,
    sampler: Sampler,
    /// 是否已执行过首次 chat prefill（对话格式管理）
    chat_prefilled: bool,
    /// 对话历史（用于构建 chat template）
    chat_history: Vec<(String, String)>, // (user, assistant) 轮次
}

impl<'a> Engine<'a> {
    /// 从 GGUF 文件加载引擎（解析超参、构建模型、加载分词器）。
    pub fn new(file: &'a GgufFile, sampler_config: SamplerConfig) -> GgufResult<Self> {
        let model = LlamaModel::new(file)?;
        let tokenizer = Tokenizer::from_gguf(file)?;
        let sampler = Sampler::new(sampler_config);
        Ok(Self {
            file,
            model,
            tokenizer,
            sampler,
            chat_prefilled: false,
            chat_history: Vec::new(),
        })
    }

    /// 分词器只读访问。
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// 模型超参只读访问。
    pub fn hparams(&self) -> &crate::infer::model::hparams::HParams {
        self.model.hparams()
    }

    /// 模型可变访问（供诊断脚本直接调 forward 拿 logits）。
    pub fn model_mut(&mut self) -> &mut LlamaModel<'a> {
        &mut self.model
    }

    /// 更新采样配置（GUI 中参数实时修改时调用）。
    pub fn set_sampler_config(&mut self, config: SamplerConfig) {
        self.sampler.set_config(config);
    }

    /// 文本 → token id 列表。
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// token id 列表 → 文本。
    pub fn detokenize(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// 生成文本：对输入 prompt 逐 token 采样，直到 EOS 或 max_tokens。
    ///
    /// `on_token` 回调在每个新 token 生成后调用（可用于流式输出）。
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> GgufResult<String>
    where
        F: FnMut(u32, &str),
    {
        let prompt_ids = self.tokenizer.encode(prompt);
        if prompt_ids.is_empty() {
            return Err(GgufError::InferenceError("prompt tokenize 为空".into()));
        }

        // 预填充：一次 forward 所有 prompt tokens（返回最后一个 token 的 logits）
        let positions: Vec<i64> = (0i64..prompt_ids.len() as i64).collect();
        let mut current_logits = self.model.forward(&prompt_ids, &positions)?;
        if current_logits.is_empty() {
            return Err(GgufError::InferenceError("logits 为空".into()));
        }

        // 生成循环
        // 字节级 BPE 不能逐 token 打印 decode_token 的转义字符串（chr(33+b) 是错误映射），
        // 必须把每个 token 还原为原始字节、累积到完整 UTF-8 字符边界再 flush 给 on_token。
        let eos = self.tokenizer.eos_id;
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut pos = prompt_ids.len() as i64;
        let mut pending: Vec<u8> = Vec::new(); // 累积的原始字节（UTF-8 片段）

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            let id = self.sampler.sample(&current_logits);
            if Some(id) == eos {
                break;
            }
            output.push(id);
            self.sampler.record(id);
            // token → 原始字节（GPT-2 bytes_to_unicode 反向映射）
            // 若整个 token 字符串都是 GPT-2 映射字符，逐字节还原；
            // 否则（Qwen 私有区 token 等）直接使用 token 字符串的原始 UTF-8 字节
            if let Some(s) = self.tokenizer.decode_token(id) {
                let all_gpt2 = s.chars().all(|c| gpt2_token_to_byte(c).is_some());
                if all_gpt2 {
                    for c in s.chars() {
                        if let Some(b) = gpt2_token_to_byte(c) {
                            pending.push(b);
                        }
                    }
                } else {
                    pending.extend_from_slice(s.as_bytes());
                }
            }
            // 在完整 UTF-8 字符边界 flush（避免把多字节汉字切成半字节打印）
            while let Some(n) = utf8_complete_len(&pending) {
                let chunk = pending.drain(0..n).collect::<Vec<_>>();
                if let Ok(text) = std::str::from_utf8(&chunk) {
                    on_token(id, text);
                }
            }

            // 单 token forward（KV cache 已持有历史）
            let p = [pos];
            current_logits = self.model.forward(&[id], &p)?;
            pos += 1;
        }

        // 拼接 prompt 已 tokenize 部分 + 输出，完整解码
        let full: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(output.iter().copied())
            .collect();
        let result = self.tokenizer.decode(&full);
        Ok(result)
    }

    /// 一次性补全（不回调），返回完整生成文本。
    pub fn complete(&mut self, prompt: &str, max_tokens: usize) -> GgufResult<String> {
        self.generate(prompt, max_tokens, |_, _| {})
    }

    /// 可取消的 prompt 补全。
    ///
    /// `on_token` 返回 `false` 时中止生成，返回已生成的完整文本。
    /// 用于 GUI 等需要支持"停止生成"的场景。
    pub fn generate_cancellable<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> GgufResult<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let prompt_ids = self.tokenizer.encode(prompt);
        if prompt_ids.is_empty() {
            return Err(GgufError::InferenceError("prompt tokenize 为空".into()));
        }

        let positions: Vec<i64> = (0i64..prompt_ids.len() as i64).collect();
        let mut current_logits = self.model.forward(&prompt_ids, &positions)?;
        if current_logits.is_empty() {
            return Err(GgufError::InferenceError("logits 为空".into()));
        }

        let eos = self.tokenizer.eos_id;
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut pos = prompt_ids.len() as i64;
        let mut pending: Vec<u8> = Vec::new();

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            let id = self.sampler.sample(&current_logits);
            if Some(id) == eos {
                break;
            }
            output.push(id);
            self.sampler.record(id);
            if let Some(s) = self.tokenizer.decode_token(id) {
                let all_gpt2 = s.chars().all(|c| gpt2_token_to_byte(c).is_some());
                if all_gpt2 {
                    for c in s.chars() {
                        if let Some(b) = gpt2_token_to_byte(c) {
                            pending.push(b);
                        }
                    }
                } else {
                    pending.extend_from_slice(s.as_bytes());
                }
            }
            while let Some(n) = utf8_complete_len(&pending) {
                let chunk = pending.drain(0..n).collect::<Vec<_>>();
                if let Ok(text) = std::str::from_utf8(&chunk) {
                    if !on_token(id, text) {
                        // 中止：flush 剩余 pending 字节
                        if !pending.is_empty() {
                            if let Ok(text) = std::str::from_utf8(&pending) {
                                let _ = on_token(id, text);
                            }
                            pending.clear();
                        }
                        let full: Vec<u32> = prompt_ids
                            .iter()
                            .copied()
                            .chain(output.iter().copied())
                            .collect();
                        return Ok(self.tokenizer.decode(&full));
                    }
                }
            }

            let p = [pos];
            current_logits = self.model.forward(&[id], &p)?;
            pos += 1;
        }

        let full: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(output.iter().copied())
            .collect();
        let result = self.tokenizer.decode(&full);
        Ok(result)
    }

    /// 多轮对话生成：KV cache 在多次调用间保持（上下文自动累积）。
    ///
    /// 自动应用 Qwen2 对话模板：
    /// - 首次调用 prefill 系统提示 + user 消息 + assistant 标记
    /// - 后续调用追加 user 消息 + assistant 标记
    /// - 模型生成回复后自动记录到对话历史
    ///
    /// 与 [`generate`] 的区别：
    /// - `generate` 每次从 position 0 开始预填充（单轮/无状态）；
    /// - `chat` 仅对**新增** token 做增量 forward，历史 K/V 复用，
    ///   适合交互问答场景（每轮输入追加到上下文）。
    ///
    /// 上下文 token 总数超过 `context_length` 时返回错误，
    /// 调用方可用 [`reset`](Engine::reset) 清空重新开始。
    ///
    /// `on_token` 回调在每个新生成 token 的完整 UTF-8 边界处调用（流式输出）。
    pub fn chat<F>(&mut self, text: &str, max_tokens: usize, mut on_token: F) -> GgufResult<String>
    where
        F: FnMut(u32, &str),
    {
        // 构建对话格式 token（Qwen2 chat template）
        // 特殊 token id（从 GGUF 词表确认）：
        // im_start=151644, im_end=151645, user=872, assistant=77091, system=8948
        let im_start = 151644u32;
        let im_end = 151645u32;
        let tok_user = 872u32;
        let tok_assistant = 77091u32;
        let tok_system = 8948u32;

        let mut new_tokens: Vec<u32> = Vec::new();
        if self.chat_prefilled {
            // 后续轮次：<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
            new_tokens.push(im_start);
            new_tokens.push(tok_user);
            // "user" 后面的 \n 由 Qwen tokenizer 编码
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(text));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_assistant);
            new_tokens.extend(self.tokenizer.encode("\n"));
        } else {
            // 首次：<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
            new_tokens.push(im_start);
            new_tokens.push(tok_system);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(
                "You are a helpful assistant.",
            ));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_user);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(text));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_assistant);
            new_tokens.extend(self.tokenizer.encode("\n"));
        }
        self.chat_prefilled = true;

        let ids = new_tokens;
        if ids.is_empty() {
            return Err(GgufError::InferenceError("chat 输入 tokenize 为空".into()));
        }
        let ctx_limit = self.hparams().context_length as usize;
        if self.model.cache_len() + ids.len() > ctx_limit {
            return Err(GgufError::InferenceError(format!(
                "上下文超出: 已有 {} tokens + 新增 {} > 上限 {}",
                self.model.cache_len(),
                ids.len(),
                ctx_limit
            )));
        }

        // 增量 forward 新增 token（positions 从 cache 长度处继续）
        let start_pos = self.model.cache_len() as i64;
        let positions: Vec<i64> = (start_pos..start_pos + ids.len() as i64).collect();
        let mut current_logits = self.model.forward_cached(&ids, &positions)?;
        if current_logits.is_empty() {
            return Err(GgufError::InferenceError("logits 为空".into()));
        }

        // 采样循环（与 generate 相同的 UTF-8 字节累积逻辑）
        // 重置 repeat penalty 计数后，将对话历史 token 加入 seen（防止跨轮重复）
        self.sampler.reset();
        for (user_msg, assistant_msg) in &self.chat_history {
            for id in self.tokenizer.encode(user_msg) {
                self.sampler.record(id);
            }
            for id in self.tokenizer.encode(assistant_msg) {
                self.sampler.record(id);
            }
        }
        let eos = self.tokenizer.eos_id;
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut pending: Vec<u8> = Vec::new();

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            let id = self.sampler.sample(&current_logits);
            if Some(id) == eos {
                break;
            }
            output.push(id);
            self.sampler.record(id);
            if let Some(s) = self.tokenizer.decode_token(id) {
                for c in s.chars() {
                    if let Some(b) = gpt2_token_to_byte(c) {
                        pending.push(b);
                    } else {
                        // 非 GPT-2 映射字符（CJK 等特殊 token）：直接追加原始 UTF-8 字节
                        let mut buf = [0u8; 4];
                        pending.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            while let Some(n) = utf8_complete_len(&pending) {
                let chunk = pending.drain(0..n).collect::<Vec<_>>();
                if let Ok(text) = std::str::from_utf8(&chunk) {
                    on_token(id, text);
                }
            }
            let p = [self.model.cache_len() as i64];
            current_logits = self.model.forward_cached(&[id], &p)?;
        }

        // 解码模型回复，记录到对话历史
        let reply = self.tokenizer.decode(&output);
        self.chat_history.push((text.to_string(), reply.clone()));
        // 回复后追加 im_end 标记（让模型知道回复结束，下一轮才能正确接上）
        let end_pos = self.model.cache_len() as i64;
        let end_positions = vec![end_pos];
        let _ = self.model.forward_cached(&[151645u32], &end_positions);

        Ok(reply)
    }

    /// 可取消的多轮对话。
    ///
    /// `on_token` 返回 `false` 时中止生成，返回已生成的完整文本。
    /// 与 [`chat`] 逻辑相同，但支持提前终止。
    pub fn chat_cancellable<F>(
        &mut self,
        text: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> GgufResult<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let im_start = 151644u32;
        let im_end = 151645u32;
        let tok_user = 872u32;
        let tok_assistant = 77091u32;
        let tok_system = 8948u32;

        let mut new_tokens: Vec<u32> = Vec::new();
        if self.chat_prefilled {
            // 后续轮次：
            new_tokens.push(im_start);
            new_tokens.push(tok_user);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(text));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_assistant);
            new_tokens.extend(self.tokenizer.encode("\n"));
        } else {
            // 首次：
            new_tokens.push(im_start);
            new_tokens.push(tok_system);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(
                "You are a helpful assistant.",
            ));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_user);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.extend(self.tokenizer.encode(text));
            new_tokens.push(im_end);
            new_tokens.extend(self.tokenizer.encode("\n"));
            new_tokens.push(im_start);
            new_tokens.push(tok_assistant);
            new_tokens.extend(self.tokenizer.encode("\n"));
        }
        self.chat_prefilled = true;

        let ids = new_tokens;
        if ids.is_empty() {
            return Err(GgufError::InferenceError("chat 输入 tokenize 为空".into()));
        }
        let ctx_limit = self.hparams().context_length as usize;
        if self.model.cache_len() + ids.len() > ctx_limit {
            return Err(GgufError::InferenceError(format!(
                "上下文超出: 已有 {} tokens + 新增 {} > 上限 {}",
                self.model.cache_len(),
                ids.len(),
                ctx_limit
            )));
        }

        let start_pos = self.model.cache_len() as i64;
        let positions: Vec<i64> = (start_pos..start_pos + ids.len() as i64).collect();
        let mut current_logits = self.model.forward_cached(&ids, &positions)?;
        if current_logits.is_empty() {
            return Err(GgufError::InferenceError("logits 为空".into()));
        }

        self.sampler.reset();
        for (user_msg, assistant_msg) in &self.chat_history {
            for id in self.tokenizer.encode(user_msg) {
                self.sampler.record(id);
            }
            for id in self.tokenizer.encode(assistant_msg) {
                self.sampler.record(id);
            }
        }
        let eos = self.tokenizer.eos_id;
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut pending: Vec<u8> = Vec::new();
        let mut cancelled = false;

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            let id = self.sampler.sample(&current_logits);
            if Some(id) == eos {
                break;
            }
            output.push(id);
            self.sampler.record(id);
            if let Some(s) = self.tokenizer.decode_token(id) {
                let all_gpt2 = s.chars().all(|c| gpt2_token_to_byte(c).is_some());
                if all_gpt2 {
                    for c in s.chars() {
                        if let Some(b) = gpt2_token_to_byte(c) {
                            pending.push(b);
                        }
                    }
                } else {
                    pending.extend_from_slice(s.as_bytes());
                }
            }
            while let Some(n) = utf8_complete_len(&pending) {
                let chunk = pending.drain(0..n).collect::<Vec<_>>();
                if let Ok(text) = std::str::from_utf8(&chunk) {
                    if !on_token(id, text) {
                        cancelled = true;
                        break;
                    }
                }
            }
            if cancelled {
                break;
            }
            let p = [self.model.cache_len() as i64];
            current_logits = self.model.forward_cached(&[id], &p)?;
        }

        let reply = self.tokenizer.decode(&output);
        self.chat_history.push((text.to_string(), reply.clone()));
        if !cancelled {
            let end_pos = self.model.cache_len() as i64;
            let end_positions = vec![end_pos];
            let _ = self.model.forward_cached(&[im_end], &end_positions);
        }

        Ok(reply)
    }

    /// 清空 KV cache 与采样器历史，开始新的对话。
    pub fn reset(&mut self) {
        self.model.reset_cache();
        self.sampler.reset();
        self.chat_prefilled = false;
        self.chat_history.clear();
    }
}

/// 返回 `bytes` 前缀中最长的完整 UTF-8 序列字节数。
///
/// 字节级 BPE 解码时，一个 token 可能只贡献一个多字节 UTF-8 字符的一部分
/// （如中文占 3 字节），需累积到完整字符边界再输出，避免打印半字节乱码。
/// 若前缀当前不构成完整序列（字符被截断），返回 `None`。
fn utf8_complete_len(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }
    // 从头逐字符累计长度，直到遇到不完整的尾字符
    let mut len = 0usize;
    while len < bytes.len() {
        let b = bytes[len];
        let need = if b < 0x80 {
            1
        } else if b & 0xE0 == 0xC0 {
            2
        } else if b & 0xF0 == 0xE0 {
            3
        } else if b & 0xF8 == 0xF0 {
            4
        } else {
            // 孤立的 continuation byte（非 UTF-8 起始字节）：丢弃该字节，
            // 返回已完整前缀 + 1，避免调用方 drain(0) 死循环。
            return Some(len + 1);
        };
        if len + need > bytes.len() {
            // 字符被截断：返回当前已完整的部分（可能为 0）
            return if len > 0 { Some(len) } else { None };
        }
        len += need;
    }
    Some(len)
}
