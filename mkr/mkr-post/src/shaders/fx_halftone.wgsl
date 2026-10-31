// halftone (mkmmd.framefx.halftone; FrameFx._halftone): comic print. `halftone` lifts the darks toward white and
// screens them with round ink dots on a turned grid; `halftone_plates` prints the red plate from a copy shifted one way
// and the blue from one shifted the other (fx/halftone.rs warps them), then mixes the print in by the strength.

struct Params {
    w: u32,
    h: u32,
    shifted: u32,       // 1: the plates are off register
    mix: u32,           // 1: mixed in by the strength (s < 1)
    level: f32,         // float32 of the level, cos, sin, the cell (px), DOT x cell, lift
    ca: f32,
    sa: f32,
    cell: f32,
    dot: f32,
    lift: f32,
    s: f32,             // the strength and 1 - it
    s1: f32,
    ink: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> red: array<vec4<f32>>;    // the print shifted for its red plate
@group(0) @binding(4) var<storage, read> blue: array<vec4<f32>>;   // and for its blue

@compute @workgroup_size(16, 16)
fn halftone(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let img = src[i].xyz;
    let l = clamp(luma(img), 0.0, 1.0);
    let d = clamp((P.level - l) / P.level, 0.0, 1.0);
    let xx = f32(id.x);
    let yy = f32(id.y);
    var u = (xx * P.ca + yy * P.sa) / P.cell;
    var v = (yy * P.ca - xx * P.sa) / P.cell;
    u = u - round(u);
    v = v - round(v);
    let dist = sqrt(u * u + v * v) * P.cell;
    var cov = clamp(P.dot * sqrt(d) - dist + 0.5, 0.0, 1.0);
    if (d <= 0.0) {
        cov = 0.0;
    }
    let base = img + (1.0 - img) * (P.lift * d);
    dst[i] = vec4<f32>(base * (1.0 - cov) + P.ink.xyz * cov, 1.0);
}

// `dst` the print (in place): its red and blue plates from the shifted copies, then the strength's mix with `src`.
@compute @workgroup_size(16, 16)
fn halftone_plates(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var out = dst[i].xyz;
    if (P.shifted == 1u) {
        out = vec3<f32>(red[i].x, out.y, blue[i].z);
    }
    if (P.mix == 1u) {
        out = src[i].xyz * P.s1 + out * P.s;
    }
    dst[i] = vec4<f32>(out, src[i].w);
}
