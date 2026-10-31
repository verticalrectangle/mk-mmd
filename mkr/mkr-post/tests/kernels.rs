//! Every kernel parses and validates as WGSL (with the shared prelude), so a broken shader fails here, not on the GPU:
//! the core passes and every frame effect's shaders/fx_*.wgsl, after the libraries it names on a `// needs FILE` line
//! (as its module puts them before it).

const COMMON: &str = include_str!("../src/shaders/common.wgsl");
const KERNELS: &[(&str, &str)] = &[
    ("unpack", include_str!("../src/shaders/unpack.wgsl")),
    ("composite", include_str!("../src/shaders/composite.wgsl")),
    ("grade", include_str!("../src/shaders/grade.wgsl")),
    ("quantize", include_str!("../src/shaders/quantize.wgsl")),
    ("warp", include_str!("../src/shaders/warp.wgsl")),
];

fn validate(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(&format!("{COMMON}\n{src}"))
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
}

#[test]
fn every_kernel_parses_and_validates() {
    for (name, src) in KERNELS {
        validate(name, src);
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut fx = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("fx_") && name.ends_with(".wgsl") {
            let src = std::fs::read_to_string(&path).unwrap();
            let mut full = String::new();
            for lib in src.lines().filter_map(|l| l.strip_prefix("// needs ")) {
                full += &std::fs::read_to_string(dir.join(lib.trim())).unwrap();
                full += "\n";
            }
            validate(&name, &(full + &src));
            fx += 1;
        }
    }
    eprintln!("{fx} frame effect kernels");
}
