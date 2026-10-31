//! Rewrite `src/lut.bin` from `mkr_shade::lut::compute()` (after changing the integrator or the sheen table):
//!
//!   cargo run --release -p mkr-shade --example write_lut

fn main() -> std::io::Result<()> {
    let t = std::time::Instant::now();
    let lut = mkr_shade::lut::compute();
    let bytes: Vec<u8> = lut.data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/lut.bin");
    std::fs::write(path, &bytes)?;
    println!("wrote {path}: {} bytes, computed in {:.2} s", bytes.len(), t.elapsed().as_secs_f64());
    Ok(())
}
