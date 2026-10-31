//! Compile every material of a scene and validate the WGSL with naga.
//!
//!   cargo run -p mkr-shade --example compile_scene -- BAKE_DIR/bake.json | PROBE.json [--dump DIR]
//!
//! A bake.json (v5, "lit" key) or a scene probe dump (dump_scene.py). Prints one line per material and a summary of
//! unsupported node types; exits 1 when a module fails validation.

use std::collections::BTreeMap;

use mkr_shade::{CompiledMaterial, GroupDesc, ImageDesc, MaterialDesc};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| anyhow::anyhow!("usage: compile_scene SCENE.json [--dump DIR]"))?;
    let dump = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1)).cloned();
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let (materials, groups, images): (BTreeMap<String, MaterialDesc>, BTreeMap<String, GroupDesc>, Vec<ImageDesc>) =
        if let Some(lit) = doc.get("lit") {
            (
                serde_json::from_value(lit["materials"].clone())?,
                serde_json::from_value(lit["groups"].clone())?,
                mkr_shade::images_from_lit(lit),
            )
        } else {
            let p = mkr_shade::probe::load(&doc);
            (p.materials, p.groups, p.images)
        };
    let mut compiled: Vec<(String, CompiledMaterial)> = Vec::new();
    let mut failed = 0;
    let mut param_base = 0;
    let t0 = std::time::Instant::now();
    for (i, (name, m)) in materials.iter().enumerate() {
        let id = format!("m{i}");
        let c = mkr_shade::compile(&id, m, &groups, &images, param_base).map_err(anyhow::Error::msg)?;
        param_base += c.param_count;
        let module = format!("{}{}{}", mkr_shade::PRELUDE, mkr_shade::RENDERER_STUBS, c.wgsl);
        let ok = mkr_shade::validate(&module);
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir)?;
            std::fs::write(format!("{dir}/{id}.wgsl"), &c.wgsl)?;
        }
        println!(
            "{id:>4} {name}: {} lines, attrs {:?}, images {:?}, params {}, transparent {}, derivatives {}, {}{}",
            c.wgsl.lines().count(),
            c.attributes,
            c.images,
            c.param_count,
            c.transparent,
            c.uses_derivatives,
            if ok.is_ok() { "valid" } else { "INVALID" },
            if c.unsupported.is_empty() { String::new() } else { format!(", unsupported {:?}", c.unsupported) }
        );
        for w in &c.warnings {
            println!("       warning: {w}");
        }
        if let Err(e) = ok {
            failed += 1;
            println!("{e}");
        }
        compiled.push((name.clone(), c));
    }
    // All materials in one module, as the renderer builds its shadow any-hit module.
    let mut all = format!("{}{}", mkr_shade::PRELUDE, mkr_shade::RENDERER_STUBS);
    for (_, c) in &compiled {
        all.push_str(&c.wgsl);
    }
    let together = mkr_shade::validate(&all);
    println!(
        "{} materials compiled in {:.1} ms; {} invalid; all-in-one module {} ({} KB)",
        compiled.len(),
        t0.elapsed().as_secs_f64() * 1e3,
        failed,
        if together.is_ok() { "valid" } else { "INVALID" },
        all.len() / 1024
    );
    if let Err(e) = together {
        println!("{e}");
        failed += 1;
    }
    let report = mkr_shade::unsupported_report(&compiled);
    if report.is_empty() {
        println!("unsupported: none");
    }
    for (what, mats) in report {
        println!("unsupported {what}: {} materials ({})", mats.len(), mats.join(", "));
    }
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
