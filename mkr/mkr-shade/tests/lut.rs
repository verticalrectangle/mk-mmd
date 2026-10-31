//! The precomputed LUT the renderer uploads is what the integrator makes.

#[test]
fn precomputed_lut_matches_the_integrator() {
    let t = std::time::Instant::now();
    let stored = mkr_shade::lut();
    eprintln!("lut(): {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let fresh = mkr_shade::lut::compute();
    assert_eq!((stored.size, stored.layers, stored.data.len()), (fresh.size, fresh.layers, fresh.data.len()));
    let worst = stored.data.iter().zip(&fresh.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1e-6, "src/lut.bin differs from lut::compute() by {worst}: run the write_lut example");
}
