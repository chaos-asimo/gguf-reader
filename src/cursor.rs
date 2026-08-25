use crate::error::{GgufError, GgufResult};

/// 带边界保护的序读取器（小端字节序）。
///
/// 所有读取操作前都会校验剩余字节数，不足时返回 [`GgufError::OutOfBounds`]，
/// 从而保证对截断/损坏文件不 panic。
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    /// 当前读取位置。
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 剩余未读字节数。
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// 底层缓冲的总长度（文件/缓冲大小）。
    pub fn total_len(&self) -> u64 {
        self.data.len() as u64
    }

    /// 读取 `n` 字节原始数据。
    fn read_bytes(&mut self, n: usize) -> GgufResult<&'a [u8]> {
        let offset = self.pos as u64;
        if n > self.remaining() {
            return Err(GgufError::OutOfBounds {
                offset,
                required: n as u64,
                file_size: self.total_len(),
            });
        }
        let chunk = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(chunk)
    }

    pub fn u8(&mut self) -> GgufResult<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    pub fn i8(&mut self) -> GgufResult<i8> {
        let b = self.read_bytes(1)?;
        Ok(b[0] as i8)
    }

    pub fn bool(&mut self) -> GgufResult<bool> {
        let b = self.read_bytes(1)?;
        Ok(b[0] != 0)
    }

    pub fn u16(&mut self) -> GgufResult<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn i16(&mut self) -> GgufResult<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> GgufResult<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i32(&mut self) -> GgufResult<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn f32(&mut self) -> GgufResult<f32> {
        let b = self.read_bytes(4)?;
        let bits = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        Ok(f32::from_bits(bits))
    }

    pub fn u64(&mut self) -> GgufResult<u64> {
        let b = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_le_bytes(arr))
    }

    pub fn i64(&mut self) -> GgufResult<i64> {
        let b = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(i64::from_le_bytes(arr))
    }

    pub fn f64(&mut self) -> GgufResult<f64> {
        let b = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        let bits = u64::from_le_bytes(arr);
        Ok(f64::from_bits(bits))
    }

    /// 读取 GGUF 字符串：`uint64` 长度前缀 + UTF-8 字节（无 null 终止）。
    ///
    /// 长度超出剩余字节返回 `OutOfBounds`；UTF-8 解码失败返回 `InvalidStringLength`。
    pub fn string(&mut self) -> GgufResult<String> {
        let len = self.u64()? as usize;
        let offset = self.pos as u64;
        if len > self.remaining() {
            return Err(GgufError::OutOfBounds {
                offset,
                required: len as u64,
                file_size: self.total_len(),
            });
        }
        let bytes = self.read_bytes(len)?;
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => Err(GgufError::InvalidStringLength(len as u64)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_scalars() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&42u8.to_le_bytes());
        buf.extend_from_slice(&(-7i8).to_le_bytes());
        buf.extend_from_slice(&0x1234u16.to_le_bytes());
        buf.extend_from_slice(&(-5678i16).to_le_bytes());
        buf.extend_from_slice(&0xdeadbeef_u32.to_le_bytes());
        buf.extend_from_slice(&(-123456i32).to_le_bytes());
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&0x0102030405060708_u64.to_le_bytes());
        buf.extend_from_slice(&(-9876543210123i64).to_le_bytes());
        buf.extend_from_slice(&2.5f64.to_le_bytes());
        buf.extend_from_slice(&1i8.to_le_bytes()); // bool true

        let mut c = Cursor::new(&buf);
        assert_eq!(c.u8().unwrap(), 42);
        assert_eq!(c.i8().unwrap(), -7);
        assert_eq!(c.u16().unwrap(), 0x1234);
        assert_eq!(c.i16().unwrap(), -5678);
        assert_eq!(c.u32().unwrap(), 0xdeadbeef);
        assert_eq!(c.i32().unwrap(), -123456);
        assert_eq!(c.f32().unwrap(), 1.5f32);
        assert_eq!(c.u64().unwrap(), 0x0102030405060708);
        assert_eq!(c.i64().unwrap(), -9876543210123);
        assert_eq!(c.f64().unwrap(), 2.5f64);
        assert!(c.bool().unwrap());
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn test_read_string() {
        let mut buf = Vec::new();
        let s = "hello";
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());

        let mut c = Cursor::new(&buf);
        assert_eq!(c.string().unwrap(), "hello");
    }

    #[test]
    fn test_empty_string() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut c = Cursor::new(&buf);
        assert_eq!(c.string().unwrap(), "");
    }

    #[test]
    fn test_out_of_bounds() {
        let buf = [1u8, 2, 3];
        let mut c = Cursor::new(&buf);
        assert!(matches!(c.u64(), Err(GgufError::OutOfBounds { .. })));
    }

    #[test]
    fn test_string_oob() {
        // 声称长度 100，但只有 3 字节
        let mut buf = Vec::new();
        buf.extend_from_slice(&100u64.to_le_bytes());
        buf.extend_from_slice(&[1, 2, 3]);
        let mut c = Cursor::new(&buf);
        assert!(matches!(c.string(), Err(GgufError::OutOfBounds { .. })));
    }

    #[test]
    fn test_invalid_utf8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&[0xff, 0xfe, 0xfd]); // 非法 UTF-8
        let mut c = Cursor::new(&buf);
        assert!(matches!(c.string(), Err(GgufError::InvalidStringLength(_))));
    }
}
