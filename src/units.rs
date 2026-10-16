//! Byte-size parsing and formatting (binary units: 1K = 1024).

pub fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    let num: f64 = num.parse().map_err(|_| format!("invalid size `{s}`"))?;
    let mult: f64 = match suffix.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1.0,
        "K" | "KB" | "KIB" => 1024.0,
        "M" | "MB" | "MIB" => 1024.0 * 1024.0,
        "G" | "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown size suffix `{other}` (use K, M or G)")),
    };
    let bytes = (num * mult) as usize;
    if bytes == 0 {
        return Err("size must be > 0".into());
    }
    Ok(bytes)
}

pub fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if v.fract() == 0.0 { format!("{v:.0} {}", UNITS[u]) } else { format!("{v:.2} {}", UNITS[u]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        assert_eq!(parse_size("4K").unwrap(), 4096);
        assert_eq!(parse_size("1.5M").unwrap(), 1536 * 1024);
        assert_eq!(parse_size("2GiB").unwrap(), 2 << 30);
        assert_eq!(parse_size("100").unwrap(), 100);
        assert!(parse_size("1X").is_err());
    }

    #[test]
    fn format() {
        assert_eq!(fmt_bytes(1 << 30), "1 GiB");
        assert_eq!(fmt_bytes(1536), "1.50 KiB");
    }
}
