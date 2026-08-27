fn main() {
    let v: f32 = 0.5;
    let bits = v.to_bits();
    println!("v={v}, bits=0x{:08X}", bits);
    let sign = ((bits & 0x8000_0000) >> 16) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    println!(
        "sign=0x{:04X}, exp={} (0x{:02X}), mant=0x{:06X}",
        sign, exp, exp, mant
    );
    let out: u16;
    if exp == 0xFF {
        out = if mant != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        };
    } else if exp > 0x8E {
        out = sign | 0x7C00;
    } else if exp >= 0x7F {
        let e = (exp - 0x7F + 15) as u16;
        let m10 = (mant >> 13) as u16;
        out = sign | (e << 10) | m10;
        println!("normal: e={}, m10=0x{:03X}, out=0x{:04X}", e, m10, out);
    } else {
        let e = (0x7F - exp) as i32;
        let m = mant | 0x800000;
        if e > 15 {
            out = sign;
        } else {
            let shift = (13 + e) as usize;
            let rounded = ((m >> shift) + if (m >> (shift - 1)) & 1 == 1 { 1 } else { 0 }) as u16;
            if rounded == 0 {
                out = sign
            } else {
                out = (rounded << 10) | sign
            };
            println!(
                "subnormal: e={}, shift={}, m=0x{:06X}, rounded={}, out=0x{:04X}",
                e, shift, m, rounded, out
            );
        }
    }
    let bytes = out.to_le_bytes();
    println!(
        "out=0x{:04X}, bytes=[0x{:02X}, 0x{:02X}]",
        out, bytes[0], bytes[1]
    );

    // Now decode
    let b: u16 = u16::from_le_bytes(bytes);
    let dsign = (b & 0x8000) != 0;
    let dexp = (b >> 10) & 0x1F;
    let dman = b & 0x3FF;
    let dv: f32 = if dexp == 0 {
        if dman == 0 {
            0.0
        } else {
            (dman as f32 / 1024.0) * 2f32.powi(-14)
        }
    } else if dexp == 31 {
        if dman == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + dman as f32 / 1024.0) * 2f32.powi(dexp as i32 - 15)
    };
    let dv = if dsign { -dv } else { dv };
    println!(
        "decoded: sign={}, exp={}, man={}, val={}",
        dsign, dexp, dman, dv
    );
}
